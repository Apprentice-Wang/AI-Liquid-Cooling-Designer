"""Run the complete AI-Liquid-Cooling-Designer generation pipeline."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Physics-based calculation", "main.py"),
    ("Sensitivity analysis", "src/sensitivity_analysis.py"),
    ("Design space search", "src/design_space_search.py"),
    ("Surrogate model training", "src/surrogate_model.py"),
    ("AI recommendation and physics verification", "src/design_recommender.py"),
    ("Design report generation", "src/report_generator.py"),
]


def run_step(index: int, name: str, script: str) -> None:
    """Run one pipeline step and raise a clear error if it fails."""
    step_start = time.perf_counter()
    print(f"\n[{index}/{len(STEPS)}] Starting: {name}", flush=True)
    print(f"Command: {sys.executable} {script}", flush=True)

    if not (PROJECT_ROOT / script).is_file():
        raise RuntimeError(
            f"Pipeline step '{name}' could not start because '{script}' was not found."
        )

    try:
        subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Python executable was not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Pipeline step '{name}' failed with exit code {exc.returncode}. "
            f"Review the output above for details."
        ) from exc

    elapsed = time.perf_counter() - step_start
    print(f"[{index}/{len(STEPS)}] Completed: {name} ({elapsed:.2f} s)", flush=True)


def main() -> int:
    """Execute all project stages in dependency order."""
    pipeline_start = time.perf_counter()
    print("=" * 68, flush=True)
    print("AI-Liquid-Cooling-Designer: complete pipeline", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print("=" * 68, flush=True)

    try:
        for index, (name, script) in enumerate(STEPS, start=1):
            run_step(index, name, script)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        print("Pipeline stopped. Later steps were not run.", file=sys.stderr, flush=True)
        return 1

    elapsed = time.perf_counter() - pipeline_start
    print("\n" + "=" * 68, flush=True)
    print(f"Pipeline completed successfully in {elapsed:.2f} s.", flush=True)
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
