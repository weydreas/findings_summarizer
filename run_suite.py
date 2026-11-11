"""Run the two fixed medical benchmarks sequentially and archive both methods.

The full discharge epicrisis is substantially larger than the renal ultrasound
fixture. This script does not parallelize requests to the shared Qwen server.
"""
import argparse
import sys

from run_test import ROOT, run_one

BENCHMARKS = (
    "examples/medical_summary_synthetic.json",
    "examples/discharge_epicrisis_deidentified.json",
)


def run_suite(model_label: str = "qwen3-1.7b-q4km") -> int:
    failed = 0
    for filename in BENCHMARKS:
        print("Running benchmark:", filename, flush=True)
        try:
            folder, code = run_one(ROOT / filename, model_label=model_label)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"SETUP_FAILED: {filename}: {type(exc).__name__}", file=sys.stderr)
            failed += 1
            continue
        print(f"Saved: {folder} (exit {code})", flush=True)
        failed += int(code != 0)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run short+long LLM benchmark cases")
    parser.add_argument("--model-label", default="qwen3-1.7b-q4km",
                        help="Recorded label of the tested model")
    args = parser.parse_args()
    return run_suite(model_label=args.model_label)


if __name__ == "__main__":
    sys.exit(main())
