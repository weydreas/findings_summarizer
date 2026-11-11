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

    def test_direct_comment_is_primary_and_extract_is_comparison(self):
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
            (src, "stop"),  # OCR restoration
            ("Прямой комментарий.", "stop"),  # direct text -> summary
            (json.dumps(facts, ensure_ascii=False), "stop"),  # legacy extraction
            ("Сравнительный комментарий.", "stop"),  # legacy facts -> summary
        ]) as request:
            result = med.summarize({"document_id": "test-1", "format": "text", "content": src})
        self.assertTrue(result["ok"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["primary_method"], "direct")
        self.assertEqual(result["schema_version"], "medical-summary-v2")
        self.assertEqual(result["comment"], "Прямой комментарий.")
        self.assertEqual(result["restored_text"], src)
        self.assertTrue(result["comparison"]["ok"])
        self.assertEqual(result["comparison"]["comment"], "Сравнительный комментарий.")
        self.assertEqual(len(result["comparison"]["findings"]), 2)
        self.assertEqual(request.call_count, 4)
        direct_messages = request.call_args_list[1].args[0]
        self.assertIn(src, direct_messages[1]["content"])
        self.assertNotIn("ИЗВЛЕЧЁННЫЕ ФАКТЫ", direct_messages[1]["content"])
        legacy_messages = request.call_args_list[3].args[0]
        self.assertIn("ИЗВЛЕЧЁННЫЕ ФАКТЫ", legacy_messages[1]["content"])

    def test_comparison_failure_does_not_erase_primary_result(self):
        src = "Исследование без отклонений."
        with patch.object(med, "_request", side_effect=[
            (src, "stop"),
            ("Без существенных изменений.", "stop"),
            med.SummaryError("INVALID_FACTS_JSON"),
        ]):
            result = med.summarize({"document_id": "test-2", "content": src})
        self.assertTrue(result["ok"])
        self.assertEqual(result["comment"], "Без существенных изменений.")
        self.assertEqual(result["comparison"], {
            "ok": False, "method": "extract", "error": "INVALID_FACTS_JSON"
        })

    def test_primary_failure_is_not_hidden(self):
        src = "Исследование без отклонений."
        with patch.object(med, "_request", side_effect=[
            (src, "stop"), med.SummaryError("LOCAL_LLM_REQUEST_FAILED"),
        ]) as request:
            with self.assertRaisesRegex(med.SummaryError, "LOCAL_LLM_REQUEST_FAILED"):
                med.summarize({"document_id": "test-3", "content": src})
        self.assertEqual(request.call_count, 2)

    def test_long_document_uses_bounded_direct_chunks(self):
        long_text = "Нормальный результат обследования.\n" * 120
        with patch.object(med, "_one_direct_comment", return_value="Краткая сводка.") as direct:
            result = med.direct_comment(long_text)
        self.assertEqual(result, "Краткая сводка.")
        self.assertGreater(direct.call_count, 1)
        for call in direct.call_args_list:
            self.assertLessEqual(len(call.args[0]), med.DIRECT_MAX_CHARS)

    def test_truncated_direct_comment_retries_and_finishes(self):
        src = "Наблюдение: параметры в пределах референса. " * 30
        calls = []

        def respond(part):
            calls.append(len(part))
            if len(part) > 700:
                raise med.SummaryError("DIRECT_COMMENT_TRUNCATED")
            if "Наблюдение:" in part:
                return "Частичный клинический комментарий."
            return "Завершённый краткий комментарий."

        with patch.object(med, "_one_direct_comment", side_effect=respond):
            result = med.direct_comment(src)
        self.assertEqual(result, "Завершённый краткий комментарий.")
        self.assertGreater(len(calls), 2)
        self.assertGreater(max(calls), 700)
        self.assertLessEqual(max(calls), med.DIRECT_MAX_CHARS)

    def test_unrecoverable_short_direct_output_never_accepted(self):
        with patch.object(med, "_request", return_value=("Незавершённый фрагмент", "length")):
            with self.assertRaisesRegex(med.SummaryError, "DIRECT_COMMENT_TRUNCATED_MIN_CHUNK"):
                med.direct_comment("Сложная формулировка длиной меньше двухсот символов.")

    def test_non_token_direct_error_is_not_retried(self):
        with patch.object(med, "_request", return_value=("Текст.", "content_filter")) as request:
            with self.assertRaisesRegex(med.SummaryError, "DIRECT_COMMENT_INCOMPLETE"):
                med.direct_comment("Документ с краткой записью.")
        self.assertEqual(request.call_count, 1)

    def test_opt_in_32k_profile_uses_larger_batches(self):
        src = "Исходный медицинский текст. " * 400
        with patch.dict(med.os.environ, {"FINDINGS_CONTEXT_PROFILE": "legacy"}):
            self.assertEqual(med._source_chunk_size(), 750)
            self.assertEqual(med._direct_input_limit(), 1800)
            self.assertEqual(med._comparison_batch_limit(), 2400)
            with patch.object(med, "_one_direct_comment", return_value="Комментарий.") as request:
                self.assertEqual(med.direct_comment(src), "Комментарий.")
                legacy_calls = request.call_count
        with patch.dict(med.os.environ, {"FINDINGS_CONTEXT_PROFILE": "32k"}):
            self.assertEqual(med._source_chunk_size(), 1500)
            self.assertEqual(med._direct_input_limit(), 16000)
            self.assertEqual(med._comparison_batch_limit(), 8000)
            with patch.object(med, "_one_direct_comment", return_value="Комментарий.") as request:
                self.assertEqual(med.direct_comment(src), "Комментарий.")
                self.assertEqual(request.call_count, 1)
        self.assertGreater(legacy_calls, 1)

    def test_invalid_context_profile_rejected(self):
        with patch.dict(med.os.environ, {"FINDINGS_CONTEXT_PROFILE": "wrong"}):
            with self.assertRaisesRegex(med.SummaryError, "INVALID_CONTEXT_PROFILE"):
                med.summarize({"document_id": "profile-case", "content": "Краткое исследование."})

    def test_32k_profile_requests_larger_completion_budget(self):
        response = {"choices": [{"message": {"content": "Тест"}, "finish_reason": "stop"}]}
        with patch.object(med.httpx, "Client") as client:
            client.return_value.__enter__.return_value.post.return_value.json.return_value = response
            with patch.dict(med.os.environ, {"FINDINGS_CONTEXT_PROFILE": "32k"}):
                self.assertEqual(med._request([{"role": "user", "content": "тест"}]), ("Тест", "stop"))
            self.assertEqual(client.return_value.__enter__.return_value.post.call_args.kwargs["json"]["max_tokens"], 2048)

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

    def test_length_mismatch_retries_with_smaller_fragments(self):
        src = "Нормальные показатели крови, давление и пульс. " * 17
        stats = {
            "retry_splits": 0, "source_fallback_blocks": 0,
            "source_fallback_chars": 0, "fallback_reasons": {},
        }

        def model(messages, *, schema=None):
            block = messages[1]["content"].split("\n", 1)[1]
            if len(block) > med.MIN_CHUNK_SIZE:
                return "слишком коротко", "stop"
            return block, "stop"

        with patch.object(med, "_request", side_effect=model):
            parts = med.restore_text(src, stats=stats)
        self.assertGreater(stats["retry_splits"], 0)
        self.assertEqual(stats["source_fallback_blocks"], 0)
        self.assertEqual(" ".join(parts).split(), src.split())

    def test_final_restoration_fallback_preserves_source_and_warns(self):
        src = "Сложный лабораторный результат: калий 3,3 ммоль/л."
        facts = {
            "diagnoses": [], "findings": [], "measurements": [],
            "treatments_and_followup": [], "uncertainties": [],
        }
        with patch.object(med, "_request", side_effect=[
            ("ошибка", "stop"),  # invalid ratio, too small to split further
            ("Прямой комментарий.", "stop"),
            (json.dumps(facts, ensure_ascii=False), "stop"),
            ("Комментарий из фактов.", "stop"),
        ]):
            result = med.summarize({"document_id": "fallback-case", "content": src})
        self.assertTrue(result["ok"])
        self.assertEqual(result["restored_text"], src)
        self.assertEqual(result["comment"], "Прямой комментарий.")
        self.assertEqual(result["comparison"]["comment"], "Комментарий из фактов.")
        self.assertEqual(result["restoration"]["source_fallback_blocks"], 1)
        self.assertEqual(result["restoration"]["source_fallback_chars"], len(src))
        self.assertEqual(result["restoration"]["fallback_reasons"], {
            "RESTORATION_LENGTH_MISMATCH": 1,
        })
        self.assertTrue(any("без автоматического восстановления" in x for x in result["review_flags"]))

    def test_network_error_remains_fatal_in_fallback_mode(self):
        stats = {
            "retry_splits": 0, "source_fallback_blocks": 0,
            "source_fallback_chars": 0, "fallback_reasons": {},
        }
        with patch.object(med, "_request", side_effect=med.SummaryError("LOCAL_LLM_REQUEST_FAILED")):
            with self.assertRaisesRegex(med.SummaryError, "LOCAL_LLM_REQUEST_FAILED"):
                med.restore_text("Нормальный текст.", stats=stats)
        self.assertEqual(stats["source_fallback_blocks"], 0)

    def test_does_not_echo_document_on_error(self):
        with patch.object(med, "_request", side_effect=med.SummaryError("LOCAL_LLM_REQUEST_FAILED")):
            with self.assertRaisesRegex(med.SummaryError, "LOCAL_LLM_REQUEST_FAILED"):
                med.summarize({"document_id": "t", "content": "CONFIDENTIAL_SENSITIVE_TEXT"})


if __name__ == "__main__":
    unittest.main()
