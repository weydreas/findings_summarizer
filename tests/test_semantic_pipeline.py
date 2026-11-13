"""Offline synthetic tests for semantic_v2, no LLM, Docker or patient data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import medical_summary as med
import run_test
import semantic_pipeline as sem


SOURCE = (
    "Основное: медикаментозно обусловленная гипотония. "
    "11.03.2024 артериальное давление 55/40 мм рт. ст. "
    "MRSA не выявлен. Креатинин 117 мкмоль/л, референс 45-84."
)


class SemanticPipelineTests(unittest.TestCase):
    def test_numerical_checks_only_numbers_dates_and_units(self):
        good = sem.numerical_guard(
            "11.03.2024 креатинин 117 мкмоль/л", SOURCE)
        self.assertEqual(good, {"numbers": [], "dates": [], "units": []})
        bad = sem.numerical_guard(
            "15.03.2025 креатинин 999 нг/мл", SOURCE)
        self.assertIn("999", bad["numbers"])
        self.assertIn("2025-03-15", bad["dates"])
        self.assertIn("нг/мл", bad["units"])
        self.assertFalse(sem._has_issues(sem.numerical_guard(
            "Гипотония не купирована", SOURCE)))
        # Numerics joined to dose units are not silently skipped.
        self.assertIn("350", sem.numerical_guard("350мг", SOURCE)["numbers"])

    def test_chunk_boundaries_preserve_original_source(self):
        blocks = sem._split_source(SOURCE)
        self.assertEqual(len(blocks), 1)
        b = blocks[0]
        self.assertEqual(SOURCE[b["start"]:b["end"]], b["text"])

    def test_fresh_role_review_rejects_lost_negation_and_archives_decisions(self):
        seen = []
        def llm(messages, *, schema=None):
            self.assertEqual(len(messages), 2)
            system, user = messages[0]["content"], messages[1]["content"]
            seen.append((system, user))
            if system == sem.EXTRACT_PROMPT:
                return json.dumps({"facts": [
                    {"kind": "diagnosis", "claim": "Медикаментозно обусловленная гипотония",
                     "priority": "essential"},
                    {"kind": "finding", "claim": "Выявлен MRSA", "priority": "essential"},
                    {"kind": "measurement", "claim": "Креатинин 117 мкмоль/л",
                     "priority": "essential"},
                    {"kind": "event", "claim": "АД 55/40 мм рт. ст. 11.03.2024",
                     "priority": "essential"},
                ]}, ensure_ascii=False), "stop"
            if system == sem.REVIEW_PROMPT:
                payload = json.loads(user)
                self.assertEqual(payload["original"], SOURCE)
                self.assertEqual(len(payload["candidate_claims"]), 4)
                return json.dumps({"verdicts": [
                    {"id": "F0001", "verdict": "supported", "reason": "Указано в диагнозе"},
                    {"id": "F0002", "verdict": "contradicted", "reason": "Источник: не выявлен"},
                    {"id": "F0003", "verdict": "supported", "reason": "Указан в анализе"},
                    {"id": "F0004", "verdict": "supported", "reason": "Указано в тексте"},
                ], "omissions": []}, ensure_ascii=False), "stop"
            if system == sem.SELECT_PROMPT:
                payload = json.loads(user)
                self.assertEqual(len(payload["verified_facts"]), 3)
                self.assertFalse(any("MRSA" in f["claim"] for f in payload["verified_facts"]))
                return json.dumps({"selected_ids": ["F0001", "F0004", "F0003"]}), "stop"
            if system == sem.SYNTH_PROMPT:
                return json.dumps({
                    "summary": "11.03.2024 медикаментозно обусловленная гипотония, "
                               "АД 55/40 мм рт. ст. Креатинин 117 мкмоль/л."
                }, ensure_ascii=False), "stop"
            if system == sem.FINAL_REVIEW_PROMPT:
                payload = json.loads(user)
                self.assertEqual(set(payload), {"original_document", "draft_summary"})
                self.assertEqual(payload["original_document"], SOURCE)
                self.assertNotIn("candidate_claims", user)
                return json.dumps({"verdict": "pass", "issues": []}), "stop"
            self.fail("Unexpected LLM prompt")
        with patch.object(med, "_request", side_effect=llm), patch.object(med, "_progress"):
            result = sem.summarize("test", SOURCE)
        self.assertTrue(result["ok"], result.get("error") or result["quality_gate"])
        self.assertEqual(result["quality_gate"]["accepted_count"], 3)
        self.assertEqual(result["quality_gate"]["rejected_count"], 1)
        self.assertEqual(result["fact_registry"][1]["review_verdict"], "contradicted")
        self.assertNotIn("MRSA", result["comment"])
        self.assertTrue(result["review_required"])
        self.assertEqual(len(seen), 5)
        self.assertNotEqual(seen[0][0], seen[1][0])
        self.assertNotEqual(seen[-1][0], seen[-2][0])

    def test_fresh_semantic_review_rejects_unsupported_final_claim(self):
        with patch.object(sem, "_extract", return_value=[
            {"kind": "diagnosis", "claim": "Гипотония", "priority": "essential"}
        ]), patch.object(sem, "_review", return_value=(
            [{"id": "F0001", "verdict": "supported", "reason": "В документе"}], []
        )), patch.object(sem, "_choose", side_effect=lambda facts: facts), \
             patch.object(sem, "_json_call", side_effect=[
                 {"summary": "Гипотония."},
                 {"verdict": "fail", "issues": [
                     {"category": "omission", "detail": "Не указано событие 55/40"}]},
             ]), patch.object(med, "_progress"):
            result = sem.summarize("test", SOURCE)
        self.assertFalse(result["ok"])
        self.assertEqual(result["quality_gate"]["semantic_verdict"], "fail")
        self.assertEqual(result["quality_gate"]["semantic_issues"][0]["category"], "omission")

    def test_block_reviewer_omission_blocks_false_success(self):
        with patch.object(sem, "_extract", return_value=[
            {"kind": "diagnosis", "claim": "Гипотония", "priority": "essential"}
        ]), patch.object(sem, "_review", return_value=(
            [{"id": "F0001", "verdict": "supported", "reason": "Подтверждено"}],
            ["Пропущено падение АД до 55/40"]
        )), patch.object(sem, "_choose", side_effect=lambda facts: facts), \
             patch.object(sem, "_json_call", side_effect=[
                 {"summary": "Гипотония."}, {"verdict": "pass", "issues": []}
             ]), patch.object(med, "_progress"):
            result = sem.summarize("synthetic", SOURCE)
        self.assertFalse(result["ok"])
        self.assertEqual(result["quality_gate"]["reported_omissions"], 1)
        self.assertEqual(len(result["quality_gate"]["omission_reports"]), 1)

    def test_only_semantic_v2_requires_extended_context(self):
        with patch.dict(med.os.environ, {"FINDINGS_PIPELINE_MODE": "semantic_v2",
                                          "FINDINGS_CONTEXT_PROFILE": "legacy"}):
            with self.assertRaisesRegex(med.SummaryError, "SEMANTIC_V2_REQUIRES_32K"):
                med.summarize({"document_id": "synthetic", "content": SOURCE})
        with patch.dict(med.os.environ, {"FINDINGS_PIPELINE_MODE": "semantic_v2",
                                          "FINDINGS_CONTEXT_PROFILE": "32k"}), \
             patch.object(sem, "summarize", return_value={"ok": True}) as mock:
            self.assertTrue(med.summarize({
                "document_id": "synthetic", "content": SOURCE})["ok"])
            mock.assert_called_once_with("synthetic", SOURCE)

    def test_model_failure_keeps_partial_private_stages(self):
        with patch.object(med, "_request",
                          side_effect=med.SummaryError("LOCAL_LLM_REQUEST_FAILED")), \
             patch.object(med, "_progress"):
            result = sem.summarize("synthetic", SOURCE)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "LOCAL_LLM_REQUEST_FAILED")
        self.assertTrue(result["blocks"])
        self.assertTrue(result["review_required"])


class SemanticArchiveTests(unittest.TestCase):
    def test_private_archive_even_when_fresh_review_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "synthetic.json"
            inp.write_text(json.dumps({"document_id": "synthetic", "content": SOURCE}),
                           encoding="utf-8")
            output = {
                "ok": False, "document_id": "synthetic",
                "primary_method": "semantic", "pipeline_mode": "semantic_v2",
                "comment": "Черновик.", "restored_text": SOURCE,
                "fact_registry": [{"id": "F0001", "claim": "Гипотония"}],
                "reviews": [{"source_block_id": "B0001", "verdicts": []}],
                "selected_ids": ["F0001"],
                "blocks": [{"id": "B0001", "start": 0, "end": len(SOURCE)}],
                "quality_gate": {"automated_gate_passed": False},
            }
            process = SimpleNamespace(returncode=1,
                                      stdout=json.dumps(output, ensure_ascii=False).encode())
            with patch.object(run_test, "RESULTS_ROOT", root / "results"), \
                 patch.dict(run_test.os.environ, {
                     "FINDINGS_CONTEXT_PROFILE": "32k",
                     "FINDINGS_PIPELINE_MODE": "semantic_v2",
                 }), patch.object(run_test.subprocess, "run", return_value=process):
                folder, code = run_test.run_one(inp)
            self.assertNotEqual(code, 0)
            for name in ("blocks.json", "facts.json", "reviews.json",
                         "selection.json", "quality.json", "result.json"):
                path = folder / name
                self.assertTrue(path.exists(), name)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            meta = json.loads((folder / "run.json").read_text())
            self.assertEqual(meta["pipeline_mode"], "semantic_v2")
            self.assertFalse(meta["automated_gate_passed"])


if __name__ == "__main__":
    unittest.main()
