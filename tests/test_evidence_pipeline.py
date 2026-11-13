"""Offline synthetic regression tests for the opt-in source-grounded pipeline."""
import io
import json
import unittest
from unittest.mock import patch

import medical_summary as med
import evidence_pipeline as ev


SOURCE = (
    "Заключительный клинический диагноз\n"
    "Основное: медикаментозно обусловленная гипотония.\n"
    "Лабораторные результаты\n"
    "Креатинин 117 мкмоль/л 45-84\n"
    "Креатинин 73 мкмоль/л 45-84\n"
    "Рекомендации\n"
    "Контроль артериального давления."
)


class EvidenceTests(unittest.TestCase):
    def test_section_blocks_are_source_spans_not_model_restorations(self):
        blocks = ev.sectionize(SOURCE)
        self.assertGreaterEqual(len(blocks), 3)
        for block in blocks:
            self.assertEqual(SOURCE[block["start"]:block["end"]], block["text"])
        self.assertTrue(any("Лабораторные результаты" in b["section"] for b in blocks))

    def test_source_anchors_and_abnormal_labs_are_required(self):
        blocks = ev.sectionize(SOURCE)
        facts = []
        ev._source_anchors(SOURCE, blocks, facts, set())
        self.assertEqual(len(facts), 2)  # main diagnosis + abnormal creatinine
        self.assertEqual({f["kind"] for f in facts}, {"diagnosis", "measurement"})
        self.assertTrue(all(f["required"] for f in facts))
        self.assertEqual(next(f for f in facts if f["kind"] == "measurement")["lab_status"], "above")
        for fact in facts:
            self.assertEqual(SOURCE[fact["source_start"]:fact["source_end"]], fact["quote"])
        # 73 is within 45-84 and must not be mislabeled abnormal.
        self.assertFalse(any("Креатинин 73" in fact["quote"] for fact in facts))

    def test_invented_llm_quote_is_rejected(self):
        b = ev.sectionize(SOURCE)[0]
        self.assertIsNone(ev._quote_location(b, "Не существующий в документе диагноз"))
        self.assertIsNotNone(ev._quote_location(b, "Основное: медикаментозно обусловленная гипотония."))

    def test_gate_catches_missing_diagnosis_and_false_normal(self):
        facts = []
        ev._source_anchors(SOURCE, ev.sectionize(SOURCE), facts, set())
        failure = ev.quality_gate("Креатинин в норме [F0002]", facts, 0)
        self.assertFalse(failure["automated_gate_passed"])
        self.assertIn("MISSING_REQUIRED_FACTS", failure["issues"])
        self.assertIn("POSSIBLE_FALSE_NORMAL", failure["issues"])
        self.assertEqual(failure["missing_required_ids"], ["F0001"])

    def test_source_grounded_pipeline_with_targeted_repair(self):
        calls = {"extract": 0, "synth": 0}
        def request(messages, *, schema=None):
            if schema is ev.AUDIT_SCHEMA:
                return json.dumps({"issues": []}), "stop"
            if schema is not None:
                calls["extract"] += 1
                return json.dumps({"facts": []}), "stop"
            calls["synth"] += 1
            if calls["synth"] == 1:
                return "Госпитализация по поводу гипотонии [F0001]", "stop"
            return "Креатинин 117 мкмоль/л, выше референса 45-84 [F0002]", "stop"

        with patch.object(med, "_request", side_effect=request), \
             patch.object(med, "_progress"):
            result = ev.summarize("synthetic", SOURCE)
        self.assertTrue(result["ok"], result.get("quality_gate"))
        self.assertEqual(result["primary_method"], "evidence")
        self.assertEqual(result["schema_version"], "medical-summary-evidence-v1")
        self.assertTrue(result["review_required"])
        self.assertTrue(result["quality_gate"]["automated_gate_passed"])
        self.assertGreaterEqual(calls["extract"], 3)
        self.assertEqual(calls["synth"], 2)
        self.assertIn("Креатинин", result["comment"])

    def test_unverified_quote_is_a_hard_gate_failure(self):
        def request(messages, *, schema=None):
            if schema is ev.AUDIT_SCHEMA:
                return json.dumps({"issues": []}), "stop"
            if schema is not None:
                return json.dumps({"facts": [{
                    "kind": "diagnosis", "statement": "Выдуманный диагноз",
                    "quote": "Несуществующий диагноз", "importance": "critical",
                }]}), "stop"
            return "Основной диагноз [F0001]\nКреатинин выше референса [F0002]", "stop"
        with patch.object(med, "_request", side_effect=request), \
             patch.object(med, "_progress"):
            result = ev.summarize("synthetic", SOURCE)
        self.assertFalse(result["ok"])
        self.assertIn("REJECTED_UNVERIFIED_QUOTES", result["quality_gate"]["issues"])
        self.assertEqual(result["quality_gate"]["rejected_quote_count"], len(ev.sectionize(SOURCE)))

    def test_semantic_audit_flags_unsupported_claim_without_faking_success(self):
        facts = []
        ev._source_anchors(SOURCE, ev.sectionize(SOURCE), facts, set())
        with patch.object(med, "_request", return_value=(
            json.dumps({"issues": [{"line": 1, "code": "numeric_mismatch"}]}), "stop"
        )), patch.object(med, "_progress"):
            issues = ev.semantic_audit(["Сводка [F0001]"], facts)
        self.assertEqual(issues, [{"part": 1, "line": 1, "code": "numeric_mismatch"}])

    def test_discharge_treatment_change_is_mandatory(self):
        source = "Рекомендации\nВ настоящее время прием препаратов, снижающих давление, не показан."
        facts = []
        ev._source_anchors(source, ev.sectionize(source), facts, set())
        self.assertTrue(any(f["kind"] == "followup" and f["required"] for f in facts))

    def test_new_pipeline_is_opt_in_and_does_not_modify_default(self):
        with patch.dict(med.os.environ, {"FINDINGS_PIPELINE_MODE": "invalid"}):
            with self.assertRaisesRegex(med.SummaryError, "INVALID_PIPELINE_MODE"):
                med.summarize({"document_id": "synthetic", "content": SOURCE})
        with patch.dict(med.os.environ, {"FINDINGS_PIPELINE_MODE": "evidence_v1"}), \
             patch.object(ev, "summarize", return_value={"ok": True, "pipeline_mode": "evidence_v1"}) as wrapped:
            self.assertEqual(med.summarize({"document_id": "synthetic", "content": SOURCE})["pipeline_mode"], "evidence_v1")
            wrapped.assert_called_once()

    def test_error_leaves_private_structured_debug_artifacts(self):
        with patch.object(med, "_request", side_effect=med.SummaryError("LOCAL_LLM_REQUEST_FAILED")), \
             patch.object(med, "_progress"):
            result = ev.summarize("synthetic", SOURCE)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "LOCAL_LLM_REQUEST_FAILED")
        self.assertTrue(result["fact_registry"])
        self.assertTrue(result["sections"])
        self.assertFalse(result["quality_gate"]["automated_gate_passed"])


if __name__ == "__main__":
    unittest.main()
