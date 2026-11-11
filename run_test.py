"""Run the local-only dual-path experiment and archive both outputs outside Git.

Usage (host, with Docker installed):
    python3 run_test.py examples/medical_summary_synthetic.json

Do not put real patient documents into a public Git repository.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "test_results"


def save_private_json(path: Path, value: object) -> None:
    # Create exclusively (never overwrite another run) with owner-only access.
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_one(source_path: Path, *, model_label: str = "qwen3-1.7b-q4km") -> tuple[Path, int]:
    os.umask(0o077)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,60}", model_label):
        raise ValueError("INVALID_MODEL_LABEL")
    source = source_path.read_bytes()
    try:
        source_json = json.loads(source)
        identifier = source_json.get("document_id", "unknown")
    except (ValueError, AttributeError):
        identifier = "invalid-input"
    identifier = re.sub(r"[^A-Za-z0-9_.-]", "-", str(identifier))[:60] or "unknown"
    if RESULTS_ROOT.is_symlink():
        raise RuntimeError("test_results must not be a symlink")
    RESULTS_ROOT.mkdir(mode=0o700, exist_ok=True)
    RESULTS_ROOT.chmod(0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = RESULTS_ROOT / f"{stamp}_{identifier}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(mode=0o700)
    save_private_json(folder / "input.json", json.loads(source.decode("utf-8-sig"))
                      if source.strip() else {"_invalid_input": "empty"})

    command = ["docker", "compose", "--profile", "medical", "run",
               "--rm", "--no-deps", "-T", "-e", "FINDINGS_PROGRESS=1", "findings"]
    # Write per-stage progress to a private file while the model is running,
    # rather than keeping all diagnostics in memory until the container exits.
    print(f"Processing: {folder}", flush=True)
    print(f"Context profile: {os.environ.get('FINDINGS_CONTEXT_PROFILE', 'legacy')}", flush=True)
    print(f"Progress log: {folder / 'stderr.log'}", flush=True)
    try:
        with (folder / "stderr.log").open("xb") as log:
            os.fchmod(log.fileno(), 0o600)
            completed = subprocess.run(
                command, input=source, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=log, check=False,
            )
        exit_code = completed.returncode
        try:
            result = json.loads(completed.stdout.decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError("non-object JSON")
        except (ValueError, UnicodeDecodeError):
            result = {"ok": False, "error": "EMPTY_CLI_OUTPUT" if not completed.stdout.strip()
                      else "INVALID_CLI_JSON"}
    except KeyboardInterrupt:
        exit_code = 130
        completed = None
        result = {"ok": False, "error": "INTERRUPTED"}

    # Full result for downstream analysis, plus separate convenient A/B artifacts.
    save_private_json(folder / "result.json", result)
    direct = {
        "ok": result.get("ok", False),
        "method": "direct",
        "document_id": result.get("document_id"),
        "comment": result.get("comment"),
        "restored_text": result.get("restored_text"),
        "restoration": result.get("restoration"),
        "review_required": result.get("review_required", True),
        "review_flags": result.get("review_flags", []),
    }
    if not result.get("ok"):
        direct["error"] = result.get("error", "PROCESSING_FAILED")
    save_private_json(folder / "direct.json", direct)
    comparison = result.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {"ok": False, "method": "extract", "error": "NOT_RUN"}
    save_private_json(folder / "extract.json", comparison)
    save_private_json(folder / "run.json", {
        "document_id": result.get("document_id"),
        "timestamp_utc": stamp,
        "exit_code": exit_code,
        "primary_method": "direct",
        "comparison_method": "extract",
        "model_label": model_label,
        "context_profile": os.environ.get("FINDINGS_CONTEXT_PROFILE", "legacy"),
        "schema_version": result.get("schema_version"),
    })
    return folder, 0 if result.get("ok") and exit_code == 0 else (exit_code or 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive direct + extract Qwen results locally")
    parser.add_argument("input_json", type=Path, help="medical document JSON file")
    parser.add_argument("--model-label", default="qwen3-1.7b-q4km", help="Label to compare later model runs")
    args = parser.parse_args()
    try:
        folder, code = run_one(args.input_json, model_label=args.model_label)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"RUN_SETUP_FAILED: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Results saved: {folder}")
    print("Files: input.json, direct.json, extract.json, result.json, run.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
