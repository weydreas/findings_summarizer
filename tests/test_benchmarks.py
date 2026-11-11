"""Offline tests for the two fixed benchmark cases; do not call the LLM."""
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import medical_summary as med
import run_suite

ROOT = Path(__file__).resolve().parent.parent


class FixedBenchmarksTests(unittest.TestCase):
    def test_both_inputs_are_tracked_and_sized_for_llm(self):
        short_path, long_path = (ROOT / path for path in run_suite.BENCHMARKS)
        short = json.loads(short_path.read_text(encoding="utf-8"))
        long = json.loads(long_path.read_text(encoding="utf-8"))
        self.assertEqual(short["format"], "text")
        self.assertEqual(long["format"], "text")
        self.assertLess(len(short["content"]), 750)
        self.assertGreater(len(long["content"]), 30000)
        for case in (short, long):
            self.assertLessEqual(len(case["content"]), med.MAX_INPUT_CHARS)
            self.assertTrue(case["document_id"])
            cleaned = med.clean_source(case["content"], "text")
            self.assertGreater(len(cleaned), len(case["content"]) * 0.8)

    def test_deidentified_document_preserves_major_clinical_sections(self):
        case = json.loads((ROOT / run_suite.BENCHMARKS[1]).read_text(encoding="utf-8"))
        src = case["content"]
        for phrase in (
            "ВЫПИСНОЙ ЭПИКРИЗ",
            "Заключительный клинический диагноз",
            "Анамнез заболевания",
            "Объективные данные",
            "Результаты медицинского обследования",
            "Клинический анализ крови",
            "Медицинские вмешательства",
            "Гемодинамика стабилизирована",
            "Рекомендации эндокринолога",
        ):
            self.assertIn(phrase, src)
        self.assertIn("Фамилия, имя, отчество: [ПАЦИЕНТКА]", src)
        self.assertIn("Номер медицинской карты: [УДАЛЕНО]", src)
        self.assertIn("Регистрация по месту жительства: [АДРЕС_УДАЛЕН]", src)
        self.assertIn("Дата рождения: [УДАЛЕНА]", src)
        self.assertNotIn("http://", src)
        self.assertNotIn("https://", src)
        # Spot-check named-role masking without repeating personal names in code.
        self.assertGreaterEqual(len(re.findall(r"\[ВРАЧ_[A-ZА-Я_]+\]", src)), 6)

    def test_suite_runs_short_then_long_and_archives_each(self):
        with patch.object(run_suite, "run_one", side_effect=[
            (Path("test_results/short"), 0),
            (Path("test_results/long"), 0),
        ]) as runner:
            self.assertEqual(run_suite.run_suite("test-model"), 0)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual([call.args[0] for call in runner.call_args_list],
                         [ROOT / path for path in run_suite.BENCHMARKS])
        self.assertEqual([call.kwargs["model_label"] for call in runner.call_args_list],
                         ["test-model", "test-model"])

    def test_suite_continues_after_failed_case(self):
        with patch.object(run_suite, "run_one", side_effect=[
            (Path("test_results/short"), 1),
            (Path("test_results/long"), 0),
        ]) as runner:
            self.assertEqual(run_suite.run_suite("test-model"), 1)
        self.assertEqual(runner.call_count, 2)

    def test_indirect_large_collection_uses_complete_fact_objects(self):
        facts = {
            "diagnoses": ["Учебный диагноз"],
            "findings": [
                {
                    "site": "орган",
                    "status": "normal",
                    "description": "учебное описание " + str(i),
                    "evidence": "Исходное свидетельство " + str(i),
                }
                for i in range(65)
            ],
            "measurements": [],
            "treatments_and_followup": [],
            "uncertainties": [],
        }
        outputs = []
        def fake_request(messages):
            batch = json.loads(messages[1]["content"].split("\n", 1)[1])
            outputs.extend(batch["findings"])
            return "Частичный итог.", "stop"
        with patch.object(med, "_request", side_effect=fake_request), \
             patch.object(med, "direct_comment", return_value="Совмещённый итог.") as combine:
            comment, strategy = med.comment_from_extracted_facts(facts)
        self.assertEqual(comment, "Совмещённый итог.")
        self.assertEqual(strategy, "hierarchical")
        self.assertEqual(len(outputs), len(facts["findings"]))
        self.assertEqual(outputs, facts["findings"])
        combine.assert_called_once()


if __name__ == "__main__":
    unittest.main()
