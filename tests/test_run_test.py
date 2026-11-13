"""Local archive runner tests using a stubbed docker process; no LLM/network."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_test


class ArchiveTests(unittest.TestCase):
    def test_archives_every_run_privately_without_overwriting(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            input_file = tmp / "synthetic.json"
            input_file.write_text(json.dumps({
                "document_id": "test-renal-1",
                "format": "text",
                "content": "Выдуманный текст для теста.",
            }, ensure_ascii=False), encoding="utf-8")
            full_result = {
                "ok": True,
                "schema_version": "medical-summary-v2",
                "document_id": "test-renal-1",
                "primary_method": "direct",
                "comment": "Прямой результат.",
                "restored_text": "Текст.",
                "review_required": True,
                "review_flags": [],
                "comparison": {
                    "ok": True,
                    "method": "extract",
                    "comment": "Косвенный результат.",
                    "diagnoses": [],
                    "findings": [],
                    "measurements": [],
                    "treatments_and_followup": [],
                    "uncertainties": [],
                    "review_flags": [],
                },
            }
            docker_result = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(full_result, ensure_ascii=False).encode(),
                stderr=b"Container Created\n",
            )
            with patch.object(run_test, "RESULTS_ROOT", tmp / "test_results"), \
                 patch.object(run_test.subprocess, "run", return_value=docker_result) as docker:
                a, code1 = run_test.run_one(input_file)
                b, code2 = run_test.run_one(input_file)
            self.assertEqual((code1, code2), (0, 0))
            self.assertNotEqual(a, b)
            for folder in (a, b):
                self.assertTrue((folder / "input.json").exists())
                self.assertTrue((folder / "result.json").exists())
                self.assertTrue((folder / "run.json").exists())
                direct = json.loads((folder / "direct.json").read_text(encoding="utf-8"))
                extract = json.loads((folder / "extract.json").read_text(encoding="utf-8"))
                self.assertEqual(direct["comment"], "Прямой результат.")
                self.assertEqual(extract["comment"], "Косвенный результат.")
                run_meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
                self.assertIn("started_at_utc", run_meta)
                self.assertIn("finished_at_utc", run_meta)
                self.assertGreaterEqual(run_meta["duration_seconds"], 0)
                self.assertLessEqual(run_meta["started_at_utc"], run_meta["finished_at_utc"])
                self.assertEqual(folder.stat().st_mode & 0o777, 0o700)
                self.assertEqual((folder / "input.json").stat().st_mode & 0o777, 0o600)
                self.assertEqual((folder / "direct.json").stat().st_mode & 0o777, 0o600)
                self.assertTrue((folder / "stderr.log").exists())
                self.assertEqual((folder / "stderr.log").stat().st_mode & 0o777, 0o600)
            self.assertEqual(docker.call_count, 2)
            for call in docker.call_args_list:
                self.assertIn("FINDINGS_PROGRESS=1", call.args[0])
                self.assertTrue(hasattr(call.kwargs["stderr"], "write"))

    def test_evidence_archive_keeps_private_stages_and_failed_gate(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            source = tmp / "synthetic.json"
            source.write_text('{"document_id":"synthetic","content":"Тест."}', encoding="utf-8")
            output = {
                "ok": False, "pipeline_mode": "evidence_v1",
                "primary_method": "evidence",
                "document_id": "synthetic", "comment": "Черновик.",
                "fact_registry": [{"id": "F0001", "quote": "Тест."}],
                "sections": [{"block_id": "B0001", "start": 0, "end": 5}],
                "draft_parts": ["Черновик."],
                "quality_gate": {"automated_gate_passed": False,
                                 "issues": ["MISSING_REQUIRED_FACTS"]},
            }
            docker_result = SimpleNamespace(returncode=1, stdout=json.dumps(output).encode())
            with patch.object(run_test, "RESULTS_ROOT", tmp / "results"), \
                 patch.dict(run_test.os.environ, {"FINDINGS_PIPELINE_MODE": "evidence_v1"}), \
                 patch.object(run_test.subprocess, "run", return_value=docker_result):
                folder, code = run_test.run_one(source)
            self.assertNotEqual(code, 0)
            for path in ("facts.json", "sections.json", "quality.json", "drafts.json"):
                self.assertTrue((folder / path).exists(), path)
                self.assertEqual((folder / path).stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads((folder / "run.json").read_text())["pipeline_mode"], "evidence_v1")
            self.assertFalse(json.loads((folder / "quality.json").read_text())["automated_gate_passed"])
            self.assertEqual(json.loads((folder / "direct.json").read_text())["method"], "evidence")

    def test_docker_failure_archives_errors_not_a_false_success(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            source = tmp / "test.json"
            source.write_text('{"document_id":"broken","content":"x"}', encoding="utf-8")
            docker_result = SimpleNamespace(returncode=1, stdout=b"", stderr=b"error\n")
            with patch.object(run_test, "RESULTS_ROOT", tmp / "test_results"), \
                 patch.object(run_test.subprocess, "run", return_value=docker_result):
                folder, code = run_test.run_one(source)
            self.assertEqual(code, 1)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result, {"ok": False, "error": "EMPTY_CLI_OUTPUT"})
            run_meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["exit_code"], 1)
            self.assertGreaterEqual(run_meta["duration_seconds"], 0)
            self.assertEqual(json.loads((folder / "extract.json").read_text(encoding="utf-8"))["error"], "NOT_RUN")
            self.assertTrue((folder / "stderr.log").exists())


if __name__ == "__main__":
    unittest.main()
