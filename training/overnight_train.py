"""
Overnight training script — runs the full pipeline:
1. Load 5 seasons of historical data (~6,000 games)
2. Build rolling pre-game snapshots (no look-ahead)
3. Train and backtest all models
4. Generate performance report
5. Push to GitHub

This is designed to run unattended overnight.
Estimated runtime: 45-90 minutes (API rate limiting is the bottleneck)
"""
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "data" / "overnight_run.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def run_step(name: str, cmd: list, timeout: int = 3600) -> bool:
    """Run a pipeline step, return True if successful."""
    log.info(f"\n{'='*60}")
    log.info(f"STEP: {name}")
    log.info(f"{'='*60}")
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, timeout=timeout,
            capture_output=False, text=True
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            log.info(f"✅ {name} completed in {elapsed:.0f}s")
            return True
        else:
            log.error(f"❌ {name} failed (exit {result.returncode}) in {elapsed:.0f}s")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"❌ {name} timed out after {timeout}s")
        return False
    except Exception as e:
        log.error(f"❌ {name} error: {e}")
        return False


def git_push(message: str):
    """Commit and push results to GitHub."""
    try:
        subprocess.run(["git", "add", "."], cwd=REPO_ROOT, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT, check=True)
        subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True)
        log.info(f"✅ Pushed to GitHub: {message}")
    except Exception as e:
        log.warning(f"Git push failed (non-critical): {e}")


def main():
    start_time = datetime.now()
    log.info(f"\n{'='*60}")
    log.info(f"🏀 NBA BETTING BRAIN — Overnight Training")
    log.info(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"{'='*60}\n")

    python = str(REPO_ROOT / "venv" / "bin" / "python3")
    steps_completed = []
    steps_failed = []

    # ── Step 1: Load historical data (5 seasons) ─────────────
    # ~5,000+ games from 2020-21 through 2025-26
    ok = run_step(
        "Load Historical Data (5 seasons)",
        [python, "training/historical_loader.py", "--seasons", "5"],
        timeout=5400  # 90 min — NBA API rate limits
    )
    if ok:
        steps_completed.append("historical_data")
    else:
        steps_failed.append("historical_data")
        # Try with fewer seasons
        log.warning("Retrying with 3 seasons...")
        ok = run_step(
            "Load Historical Data (3 seasons fallback)",
            [python, "training/historical_loader.py", "--seasons", "3"],
            timeout=3600
        )
        if ok:
            steps_completed.append("historical_data_3seasons")
        else:
            steps_failed.append("historical_data_fallback")

    # ── Step 2: Train models ──────────────────────────────────
    ok = run_step(
        "Model Training + Backtesting",
        [python, "training/train.py"],
        timeout=1800  # 30 min
    )
    if ok:
        steps_completed.append("model_training")
    else:
        steps_failed.append("model_training")

    # ── Step 3: Run daily update (today's games) ─────────────
    ok = run_step(
        "Daily Data Update",
        [python, "daily_run.py"],
        timeout=300
    )
    if ok:
        steps_completed.append("daily_update")

    # ── Step 4: Generate simulation report ───────────────────
    # Run backtesting simulation report if training succeeded
    if "model_training" in steps_completed:
        report_script = REPO_ROOT / "backtesting" / "simulation_report.py"
        if report_script.exists():
            ok = run_step(
                "Simulation Report",
                [python, str(report_script)],
                timeout=600
            )
            if ok:
                steps_completed.append("simulation_report")

    # ── Summary ───────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).seconds
    log.info(f"\n{'='*60}")
    log.info(f"OVERNIGHT RUN COMPLETE")
    log.info(f"  Runtime: {elapsed//60}m {elapsed%60}s")
    log.info(f"  Completed: {steps_completed}")
    log.info(f"  Failed: {steps_failed}")
    log.info(f"{'='*60}")

    # ── Push results to GitHub ────────────────────────────────
    completed_str = ", ".join(steps_completed) if steps_completed else "none"
    git_push(
        f"Overnight training {start_time.strftime('%Y-%m-%d')}\n\n"
        f"Steps: {completed_str}\n"
        f"Runtime: {elapsed//60}m {elapsed%60}s"
    )

    return len(steps_failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
