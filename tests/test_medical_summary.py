"""Offline checks; no real medical records, model weights, network or Docker required."""
import json
import unittest
from unittest.mock import patch

import medical_summary as med


class MedicalSummaryTests(unittest.TestCase):
    def test_html_parser_preserves_multiple_organs_and_removes_scripts(self):
        html = '<html><body><p>Диагноз: А</p><table><tr><td>Почки</td><td>Без изменений</td></tr></table><p>Печень: изменена</p><script>IGNORE</script></body></html>'
        actual = med.clean_source(html, "auto")
        for token in ("Диагноз: А", "Почки", "Без изменений", "Печень: изменена"):
            self.assertIn(token, actual)
        self.assertNotIn("IGNORE", actual)

    def test_split_does_not_drop_words(self):
        src = "Пример " * 170
        blocks = med.split_chunks(src, 90)
        self.assertGreater(len(blocks), 2)
        self.assertEqual(" ".join(blocks), src.strip())

    def test_full_pipeline_uses_same_llm_and_returns_verified_shape(self):
        src = "УЗИ почек. Основной диагноз: нефрит. Почки без изменений. Печень увеличена."
        facts = {
            "diagnoses": ["нефрит"],
            "findings": [
                {"site": "почки", "status": "normal", "description": "без изменений", "evidence": "Почки без изменений."},
                {"site": "печень", "status": "abnormal", "description": "увеличена", "evidence": "Печень увеличена."},
            ],
            "measurements": [],
            "treatments_and_followup": [],
            "uncertainties": [],
        }
        with patch.object(med, "_request", side_effect=[
            (src, "stop"),
            (json.dumps(facts, ensure_ascii=False), "stop"),
            ("Нефрит. Почки без изменений. Печень увеличена.", "stop"),
        ]) as request:
            result = med.summarize({"document_id": "test-1", "format": "text", "content": src})
        self.assertTrue(result["ok"])
        self.assertTrue(result["review_required"])
        self.assertEqual(len(result["findings"]), 2)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(result["restored_text"], src)

    def test_rejects_invalid_schema_and_bad_identifiers(self):
        with self.assertRaises(med.SummaryError):
            med._validate_facts({"diagnoses": []})
        with self.assertRaises(med.SummaryError):
            med.summarize({"document_id": "../bad", "content": "text"})

    def test_prevents_nonlocal_llm_endpoint(self):
        with patch.dict(med.os.environ, {"LLM_URL": "https://external.example/v1/chat/completions"}):
            with self.assertRaisesRegex(med.SummaryError, "LOCAL_LLM_URL_REQUIRED"):
                med._request([{"role": "user", "content": "dummy"}])

    def test_incomplete_restoration_fails_closed(self):
        with patch.object(med, "_request", return_value=("обрезанный", "length")):
            with self.assertRaisesRegex(med.SummaryError, "RESTORATION_TRUNCATED"):
                med.restore_text("Текст " * 10)

    def test_does_not_echo_document_on_error(self):
        with patch.object(med, "_request", side_effect=med.SummaryError("LOCAL_LLM_REQUEST_FAILED")):
            with self.assertRaisesRegex(med.SummaryError, "LOCAL_LLM_REQUEST_FAILED"):
                med.summarize({"document_id": "t", "content": "CONFIDENTIAL_SENSITIVE_TEXT"})


if __name__ == "__main__":
    unittest.main()
