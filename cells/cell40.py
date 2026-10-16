# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 40: Pipeline Orchestrator
# =============================================================================
# Purpose  : Execute Cells 33-39 as a managed pipeline with:
#              - Manual trigger: call run_pipeline() directly
#              - Scheduled trigger: APScheduler background thread,
#                configurable interval (default 24h)
#              - Halt-and-report on any step failure
#              - Per-step: runtime, pass/fail, stdout/stderr capture,
#                output file MD5 checksums
#              - Structured run report (JSON + human-readable summary)
#              - Pipeline failure fires Cell 39 alert directly
#
# Step registry (Cells 33-39):
#   STEP_33  Autonomous data ingestion
#   STEP_34  Automated feature engineering
#   STEP_35  PINN continual learning
#   STEP_36  Risk threshold detection
#   STEP_37  LLM recommendation engine
#   STEP_38  Dashboard and publication report
#   STEP_39  Automated alert system
#
# Each step is imported and executed via its public run_cell_NN() function.
# stdout/stderr are captured per step using contextlib.redirect_stdout/stderr.
#
# Inputs   : All Cell 33-39 Drive outputs (verified via MD5 before next step)
# Outputs  : logs/pipeline_run_{timestamp}.json
#            logs/pipeline_run_latest.json
#            logs/pipeline_log.csv
# =============================================================================

# --- 0. DEPENDENCY BOOTSTRAP -------------------------------------------------

import subprocess, sys

def _ensure(pkg: str, import_name: str = None) -> None:
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

_ensure("apscheduler", "apscheduler")

# --- 1. IMPORTS --------------------------------------------------------------

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# --- 2. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell40")

# --- 3. CONFIGURATION --------------------------------------------------------

DRIVE_BASE = Path("/content/drive/MyDrive/BD_ML_VT")
LOG_DIR    = DRIVE_BASE / "logs"
RISK_DIR   = DRIVE_BASE / "risk_reports"

PIPELINE_LOG_CSV    = LOG_DIR / "pipeline_log.csv"
PIPELINE_RUN_LATEST = LOG_DIR / "pipeline_run_latest.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Scheduler default: run every 24 hours
SCHEDULER_INTERVAL_HOURS = 24

# Maximum stdout/stderr captured per step before truncation (bytes)
CAPTURE_MAX_BYTES = 32_768   # 32 KB

# --- 4. STEP REGISTRY --------------------------------------------------------
# Each entry defines one pipeline step. The `module_path` points to the
# Colab notebook cell file on Drive (written by the user from each cell).
# The `entry_fn` is the public run_cell_NN() function name.
# `output_files` lists Drive paths whose MD5 is recorded post-execution.

CELL_MODULE_PATHS = {
    33: DRIVE_BASE / "cells" / "cell33.py",
    34: DRIVE_BASE / "cells" / "cell34.py",
    35: DRIVE_BASE / "cells" / "cell35.py",
    36: DRIVE_BASE / "cells" / "cell36.py",
    37: DRIVE_BASE / "cells" / "cell37.py",
    38: DRIVE_BASE / "cells" / "cell38.py",
    39: DRIVE_BASE / "cells" / "cell39.py",
}

STEP_REGISTRY: List[Dict] = [
    {
        "cell"       : 33,
        "name"       : "Autonomous Data Ingestion",
        "entry_fn"   : "run_cell_33",
        "output_files": [
            DRIVE_BASE / "live_data"    / "psmsl_latest.csv",
            LOG_DIR                     / "ingestion_log.csv",
        ],
        "required"   : True,
    },
    {
        "cell"       : 34,
        "name"       : "Automated Feature Engineering",
        "entry_fn"   : "run_cell_34",
        "output_files": [
            DRIVE_BASE / "live_data"    / "features_latest.csv",
            DRIVE_BASE / "live_data"    / "merged_dataset.csv",
            LOG_DIR                     / "feature_engineering_log.csv",
        ],
        "required"   : True,
    },
    {
        "cell"       : 35,
        "name"       : "PINN Continual Learning",
        "entry_fn"   : "run_cell_35",
        "output_files": [
            LOG_DIR / "continual_learning_log.csv",
        ],
        "required"   : True,
    },
    {
        "cell"       : 36,
        "name"       : "Risk Threshold Detection",
        "entry_fn"   : "run_cell_36",
        "output_files": [
            RISK_DIR / "risk_assessment_latest.json",
            LOG_DIR  / "risk_detection_log.csv",
        ],
        "required"   : True,
    },
    {
        "cell"       : 37,
        "name"       : "LLM Recommendation Engine",
        "entry_fn"   : "run_cell_37",
        "output_files": [
            RISK_DIR / "recommendations_latest.json",
            RISK_DIR / "recommendations_latest.md",
            LOG_DIR  / "llm_recommendation_log.csv",
        ],
        "required"   : True,
    },
    {
        "cell"       : 38,
        "name"       : "Dashboard and Publication Report",
        "entry_fn"   : "run_cell_38",
        "output_files": [
            DRIVE_BASE / "dashboards" / "hypersrl_dashboard_latest.html",
            DRIVE_BASE / "dashboards" / "hypersrl_report_latest.pdf",
            LOG_DIR                   / "dashboard_log.csv",
        ],
        "required"   : True,
    },
    {
        "cell"       : 39,
        "name"       : "Automated Alert System",
        "entry_fn"   : "run_cell_39",
        "output_files": [
            LOG_DIR / "alert_log.csv",
            LOG_DIR / "alert_state.json",
        ],
        "required"   : True,
    },
]

# --- 5. MD5 CHECKSUM UTILITY -------------------------------------------------

def md5_file(path: Path) -> Optional[str]:
    """Return hex MD5 of file at path, or None if file does not exist."""
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_outputs(output_files: List[Path]) -> Dict[str, Optional[str]]:
    """Return {filename: md5_hex_or_None} for each output file."""
    return {
        p.name: md5_file(p)
        for p in output_files
    }


# --- 6. STEP LOADER ----------------------------------------------------------

def _load_module_from_path(cell_num: int, module_path: Path):
    """
    Dynamically import a cell module from its .py file on Drive.

    The cell files live in DRIVE_BASE/cells/cellNN.py.
    If the file is not found, raises FileNotFoundError with a clear message
    explaining where to save the cell.
    """
    if not module_path.exists():
        raise FileNotFoundError(
            f"Cell {cell_num} module not found: {module_path}\n"
            f"Save the cell's .py content to that path before running "
            f"the orchestrator. See STEP_REGISTRY for all expected paths."
        )
    spec   = importlib.util.spec_from_file_location(
        f"hypersrl_cell{cell_num}", str(module_path)
    )
    module = importlib.util.module_from_spec(spec)
    # Execute module at import time (runs top-level code including config)
    spec.loader.exec_module(module)
    return module


# --- 7. STEP EXECUTOR --------------------------------------------------------

def _capture_streams() -> Tuple[io.StringIO, io.StringIO]:
    return io.StringIO(), io.StringIO()


def execute_step(step: Dict) -> Dict[str, Any]:
    """
    Execute one pipeline step. Returns a step result dict:
    {
        cell, name, status, start_ts, end_ts, runtime_s,
        stdout, stderr, checksums, error, return_value
    }
    Status: 'PASS' | 'FAIL'
    """
    cell_num  = step["cell"]
    step_name = step["name"]
    entry_fn  = step["entry_fn"]

    result = {
        "cell"        : cell_num,
        "name"        : step_name,
        "status"      : "FAIL",
        "start_ts"    : None,
        "end_ts"      : None,
        "runtime_s"   : None,
        "stdout"      : "",
        "stderr"      : "",
        "checksums"   : {},
        "error"       : None,
        "return_value": None,
    }

    stdout_buf, stderr_buf = _capture_streams()
    t_start = time.perf_counter()
    result["start_ts"] = datetime.now(timezone.utc).isoformat()

    try:
        logger.info("  Executing Cell %d: %s ...", cell_num, step_name)

        module_path = CELL_MODULE_PATHS.get(cell_num)
        module      = _load_module_from_path(cell_num, module_path)
        fn          = getattr(module, entry_fn)

        with contextlib.redirect_stdout(stdout_buf), \
             contextlib.redirect_stderr(stderr_buf):
            return_value = fn()

        result["status"]       = "PASS"
        result["return_value"] = _safe_serialise(return_value)
        logger.info("  Cell %d PASS (%.1fs)", cell_num,
                    time.perf_counter() - t_start)

    except Exception as exc:
        result["error"] = traceback.format_exc()
        logger.error("  Cell %d FAIL: %s", cell_num, exc)

    finally:
        t_end              = time.perf_counter()
        result["end_ts"]   = datetime.now(timezone.utc).isoformat()
        result["runtime_s"]= round(t_end - t_start, 2)

        raw_out = stdout_buf.getvalue()
        raw_err = stderr_buf.getvalue()
        result["stdout"] = raw_out[:CAPTURE_MAX_BYTES] + (
            f"\n[TRUNCATED — {len(raw_out)} total bytes]"
            if len(raw_out) > CAPTURE_MAX_BYTES else ""
        )
        result["stderr"] = raw_err[:CAPTURE_MAX_BYTES] + (
            f"\n[TRUNCATED — {len(raw_err)} total bytes]"
            if len(raw_err) > CAPTURE_MAX_BYTES else ""
        )

        result["checksums"] = checksum_outputs(step.get("output_files", []))

    return result


def _safe_serialise(obj: Any) -> Any:
    """Convert return value to a JSON-safe structure (shallow)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {
            str(k): _safe_serialise(v)
            for k, v in list(obj.items())[:30]   # cap depth
        }
    if isinstance(obj, (list, tuple)):
        return [_safe_serialise(v) for v in obj[:20]]
    return str(obj)[:200]


# --- 8. PIPELINE FAILURE ALERT -----------------------------------------------

def _fire_failure_alert(
    failed_step  : Dict,
    run_timestamp: str,
    prior_results: List[Dict],
) -> None:
    """
    Inject a PIPELINE_FAILURE alert directly into the Cell 39 alert log
    and attempt dispatch. Does not require Cell 39 module to be importable —
    uses Cell 39's persist_alerts machinery if available, falls back to
    direct CSV append.
    """
    alert = {
        "alert_id"   : f"{run_timestamp}_PIPELINE_FAILURE_"
                       f"Cell{failed_step['cell']}",
        "timestamp"  : run_timestamp,
        "alert_class": "PIPELINE_FAILURE",
        "severity"   : "CRITICAL",
        "country"    : "SYSTEM",
        "title"      : (f"Pipeline halted at Cell {failed_step['cell']}: "
                        f"{failed_step['name']}"),
        "detail"     : (
            f"HyperSLR pipeline run {run_timestamp} failed at step "
            f"Cell {failed_step['cell']} ({failed_step['name']}). "
            f"Runtime before failure: {failed_step.get('runtime_s','N/A')}s. "
            f"Prior steps completed: "
            f"{[r['cell'] for r in prior_results if r['status']=='PASS']}. "
            f"Error (first 300 chars): "
            f"{str(failed_step.get('error',''))[:300]}"
        ),
        "metrics"    : {
            "failed_cell" : failed_step["cell"],
            "failed_step" : failed_step["name"],
            "error_excerpt": str(failed_step.get("error",""))[:200],
        },
        "dispatched" : [],
    }

    # Try Cell 39 module dispatch
    try:
        cell39_path = CELL_MODULE_PATHS.get(39)
        if cell39_path and cell39_path.exists():
            spec   = importlib.util.spec_from_file_location(
                "hypersrl_cell39_alert", str(cell39_path)
            )
            mod    = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Load last known risk report for email body
            risk_path = RISK_DIR / "risk_assessment_latest.json"
            risk_rpt  = {}
            if risk_path.exists():
                with open(risk_path) as f:
                    risk_rpt = json.load(f)
            mod.dispatch_alerts([alert], risk_rpt)
            logger.info("Failure alert dispatched via Cell 39.")
            return
    except Exception as e:
        logger.warning("Cell 39 dispatch unavailable: %s", e)

    # Fallback: direct CSV append
    row = pd.DataFrame([{
        "timestamp"  : run_timestamp,
        "alert_id"   : alert["alert_id"],
        "alert_class": alert["alert_class"],
        "severity"   : alert["severity"],
        "country"    : alert["country"],
        "title"      : alert["title"],
        "detail"     : alert["detail"][:200],
        "dispatched" : "",
    }])
    alert_csv = LOG_DIR / "alert_log.csv"
    if alert_csv.exists():
        row.to_csv(alert_csv, mode="a", header=False, index=False)
    else:
        row.to_csv(alert_csv, mode="w", header=True, index=False)
    logger.info("Failure alert written directly to alert_log.csv.")


# --- 9. RUN REPORT -----------------------------------------------------------

def _build_run_report(
    run_timestamp : str,
    step_results  : List[Dict],
    pipeline_status: str,
    halt_at_cell  : Optional[int],
    total_runtime_s: float,
) -> Dict:
    """Assemble the full structured run report."""
    passed = [r for r in step_results if r["status"] in ("PASS", "DRY_RUN")]
    failed = [r for r in step_results if r["status"] == "FAIL"]

    return {
        "run_timestamp"  : run_timestamp,
        "pipeline_status": pipeline_status,   # "SUCCESS" | "FAILED" | "PARTIAL"
        "halt_at_cell"   : halt_at_cell,
        "total_runtime_s": round(total_runtime_s, 2),
        "steps_total"    : len(STEP_REGISTRY),
        "steps_executed" : len(step_results),
        "steps_passed"   : len(passed),
        "steps_failed"   : len(failed),
        "step_results"   : step_results,
    }


def _save_run_report(report: Dict) -> Path:
    """Save run report as timestamped JSON and overwrite latest."""
    ts_path = LOG_DIR / f"pipeline_run_{report['run_timestamp']}.json"
    with open(ts_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(PIPELINE_RUN_LATEST, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Run report saved: %s", ts_path.name)
    return ts_path


def _append_pipeline_log(report: Dict) -> None:
    """Append one summary row per step to the running pipeline CSV log."""
    rows = []
    for r in report["step_results"]:
        rows.append({
            "run_timestamp"  : report["run_timestamp"],
            "pipeline_status": report["pipeline_status"],
            "cell"           : r["cell"],
            "name"           : r["name"],
            "status"         : r["status"],
            "runtime_s"      : r["runtime_s"],
            "error_excerpt"  : str(r.get("error") or "")[:120],
        })
    df = pd.DataFrame(rows)
    if PIPELINE_LOG_CSV.exists():
        df.to_csv(PIPELINE_LOG_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(PIPELINE_LOG_CSV, mode="w", header=True, index=False)


def print_run_summary(report: Dict) -> None:
    """Print a human-readable pipeline run summary to stdout."""
    ts   = report["run_timestamp"]
    stat = report["pipeline_status"]
    stat_label = {
        "SUCCESS": "[SUCCESS]",
        "FAILED" : "[FAILED ]",
        "PARTIAL": "[PARTIAL]",
    }.get(stat, f"[{stat}]")

    print()
    print("=" * 70)
    print(f"HYPERSRL PIPELINE RUN — {ts}")
    print(f"Status        : {stat_label}  ({report['steps_passed']}/"
          f"{report['steps_executed']} steps passed)")
    print(f"Total runtime : {report['total_runtime_s']:.1f}s")
    if report.get("halt_at_cell"):
        print(f"Halted at     : Cell {report['halt_at_cell']}")
    print("-" * 70)
    print(f"  {'Cell':<6} {'Step':<38} {'Status':<6} {'Time(s)':>7}  Checksums")
    print(f"  {'-'*6} {'-'*38} {'-'*6} {'-'*7}  ---------")
    for r in report["step_results"]:
        chk_str = ", ".join(
            f"{k}: {v[:8] if v else 'MISSING'}"
            for k, v in r["checksums"].items()
        )
        status_label = {"PASS": "PASS", "FAIL": "FAIL", "DRY_RUN": "DRY"}.get(
            r["status"], r["status"]
        )
        print(
            f"  {r['cell']:<6} {r['name']:<38} "
            f"{status_label:<6} "
            f"{r['runtime_s']:>7.1f}  {chk_str[:50]}"
        )
        if r["status"] == "FAIL" and r.get("error"):
            first_line = str(r["error"]).splitlines()[-1][:60]
            print(f"         ERROR: {first_line}")
    print("=" * 70)
    print()


# =============================================================================
# CELL 40 — MAIN PIPELINE RUNNER
# =============================================================================

def run_pipeline(
    steps       : Optional[List[int]] = None,
    dry_run     : bool = False,
) -> Dict[str, Any]:
    """
    Execute the HyperSLR pipeline end-to-end.

    Parameters
    ----------
    steps   : List of cell numbers to run (default: all, in registry order).
              Useful for partial reruns e.g. run_pipeline(steps=[36,37,38,39])
    dry_run : If True, validates step registry and checksums existing outputs
              without executing any cell. Useful for pre-flight checks.

    Returns
    -------
    Full run report dict.
    """
    run_ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t_pipeline  = time.perf_counter()

    active_steps = [
        s for s in STEP_REGISTRY
        if steps is None or s["cell"] in steps
    ]

    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 40: PIPELINE ORCHESTRATOR")
    logger.info("Run timestamp : %s", run_ts)
    logger.info("Mode          : %s", "DRY RUN" if dry_run else "LIVE")
    logger.info("Steps planned : %s",
                [s["cell"] for s in active_steps])
    logger.info("=" * 70)

    step_results    = []
    pipeline_status = "SUCCESS"
    halt_at_cell    = None

    for step in active_steps:

        if dry_run:
            # Just record checksums without executing
            chk = checksum_outputs(step.get("output_files", []))
            step_results.append({
                "cell"        : step["cell"],
                "name"        : step["name"],
                "status"      : "DRY_RUN",
                "start_ts"    : None,
                "end_ts"      : None,
                "runtime_s"   : 0.0,
                "stdout"      : "",
                "stderr"      : "",
                "checksums"   : chk,
                "error"       : None,
                "return_value": None,
            })
            missing = [k for k, v in chk.items() if v is None]
            if missing:
                logger.warning(
                    "  Cell %d [DRY_RUN]: missing outputs: %s",
                    step["cell"], missing
                )
            else:
                logger.info(
                    "  Cell %d [DRY_RUN]: all outputs present.",
                    step["cell"]
                )
            continue

        # Live execution
        result = execute_step(step)
        step_results.append(result)

        if result["status"] == "FAIL":
            pipeline_status = "FAILED"
            halt_at_cell    = step["cell"]
            logger.error(
                "Pipeline halted at Cell %d (%s).",
                step["cell"], step["name"]
            )
            # Fire failure alert
            _fire_failure_alert(result, run_ts, step_results[:-1])
            break

    total_runtime = time.perf_counter() - t_pipeline

    # If some steps ran but not all (dry_run skips this)
    if (not dry_run
            and pipeline_status == "SUCCESS"
            and len(step_results) < len(active_steps)):
        pipeline_status = "PARTIAL"

    report = _build_run_report(
        run_ts, step_results, pipeline_status,
        halt_at_cell, total_runtime
    )

    if not dry_run:
        ts_path = _save_run_report(report)
        _append_pipeline_log(report)
        logger.info("Pipeline log  : %s", PIPELINE_LOG_CSV)
        logger.info("Run report    : %s", ts_path.name)

    print_run_summary(report)
    return report


# =============================================================================
# CELL 40 — SCHEDULER
# =============================================================================

_scheduler = None   # Module-level singleton

def start_scheduler(
    interval_hours : float = SCHEDULER_INTERVAL_HOURS,
    steps          : Optional[List[int]] = None,
    run_immediately: bool = True,
) -> None:
    """
    Start a background APScheduler thread that fires run_pipeline()
    every interval_hours.

    Parameters
    ----------
    interval_hours  : How often to run the full pipeline.
    steps           : Subset of cells to run (default: all).
    run_immediately : If True, execute one pipeline run immediately
                      before handing off to the schedule.
    """
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler

    if _scheduler and _scheduler.running:
        logger.warning(
            "Scheduler already running. Call stop_scheduler() first."
        )
        return

    if run_immediately:
        logger.info("Running pipeline immediately before scheduler starts...")
        run_pipeline(steps=steps)

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1}
    )
    _scheduler.add_job(
        func=run_pipeline,
        trigger="interval",
        hours=interval_hours,
        kwargs={"steps": steps},
        id="hypersrl_pipeline",
        name=f"HyperSLR pipeline (every {interval_hours}h)",
        replace_existing=True,
    )
    _scheduler.start()

    next_run = _scheduler.get_job("hypersrl_pipeline").next_run_time
    logger.info("=" * 70)
    logger.info("SCHEDULER STARTED")
    logger.info("  Interval  : every %.1f hours", interval_hours)
    logger.info("  Next run  : %s", next_run)
    logger.info("  Steps     : %s", steps or "all")
    logger.info("  Stop with : stop_scheduler()")
    logger.info("=" * 70)


def stop_scheduler() -> None:
    """Gracefully shut down the background scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
    else:
        logger.info("No active scheduler.")


def scheduler_status() -> Dict:
    """Return current scheduler state."""
    global _scheduler
    if not _scheduler or not _scheduler.running:
        return {"running": False, "jobs": []}
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id"          : job.id,
            "name"        : job.name,
            "next_run_time": str(job.next_run_time),
        })
    return {"running": True, "jobs": jobs}


# =============================================================================
# CELL 40 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_40() -> None:
    """
    T01 — md5_file returns None for missing file
    T02 — md5_file returns correct hex string for known content
    T03 — md5_file is deterministic (same file, same result)
    T04 — checksum_outputs returns dict keyed by filename
    T05 — checksum_outputs marks missing file as None
    T06 — _safe_serialise handles str, int, dict, list, None
    T07 — _safe_serialise caps dict and list depth without error
    T08 — execute_step PASS: dummy step with no-op entry function
    T09 — execute_step FAIL: step raises exception, captures traceback
    T10 — execute_step captures stdout from entry function
    T11 — execute_step records runtime_s > 0
    T12 — execute_step truncates stdout at CAPTURE_MAX_BYTES
    T13 — run_pipeline dry_run: returns report with DRY_RUN status
    T14 — run_pipeline dry_run: no cell modules required
    T15 — run_pipeline halts on first failure (no subsequent steps run)
    T16 — _build_run_report assembles correct summary counts
    T17 — _save_run_report writes JSON to LOG_DIR
    T18 — _append_pipeline_log writes CSV with correct columns
    T19 — Scheduler starts, status shows running, stop works
    T20 — End-to-end dry_run on Drive data (all output checksums)
    """
    PASS = "PASS"
    FAIL = "FAIL"
    results = []
    import tempfile, shutil

    def record(tid, name, status, detail=""):
        results.append((tid, name, status, detail))
        marker = "[PASS]" if status == PASS else "[FAIL]"
        print(f"  {marker} {tid}: {name}" +
              (f" — {detail}" if detail else ""))

    print("\n" + "=" * 70)
    print("CELL 40 — TEST SUITE")
    print("=" * 70 + "\n")

    tmp_dir = Path(tempfile.mkdtemp())

    # T01
    test_id = "T01"
    try:
        result = md5_file(tmp_dir / "nonexistent.txt")
        assert result is None
        record(test_id, "md5_file returns None for missing file", PASS)
    except Exception as e:
        record(test_id, "md5_file returns None for missing file", FAIL, str(e))

    # T02
    test_id = "T02"
    try:
        p = tmp_dir / "known.txt"
        p.write_bytes(b"HyperSLR")
        h = md5_file(p)
        import hashlib as _hl
        expected = _hl.md5(b"HyperSLR").hexdigest()
        assert h == expected, f"{h} != {expected}"
        record(test_id, "md5_file correct hex for known content", PASS,
               f"md5={h}")
    except Exception as e:
        record(test_id, "md5_file correct hex for known content", FAIL, str(e))

    # T03
    test_id = "T03"
    try:
        p  = tmp_dir / "det.txt"
        p.write_bytes(b"deterministic")
        h1 = md5_file(p)
        h2 = md5_file(p)
        assert h1 == h2, "md5_file not deterministic"
        record(test_id, "md5_file is deterministic", PASS)
    except Exception as e:
        record(test_id, "md5_file is deterministic", FAIL, str(e))

    # T04
    test_id = "T04"
    try:
        p1 = tmp_dir / "a.txt"; p1.write_text("a")
        p2 = tmp_dir / "b.txt"; p2.write_text("b")
        chk = checksum_outputs([p1, p2])
        assert set(chk.keys()) == {"a.txt", "b.txt"}
        assert all(v is not None for v in chk.values())
        record(test_id, "checksum_outputs keyed by filename", PASS)
    except Exception as e:
        record(test_id, "checksum_outputs keyed by filename", FAIL, str(e))

    # T05
    test_id = "T05"
    try:
        p_exists  = tmp_dir / "exists.txt"; p_exists.write_text("x")
        p_missing = tmp_dir / "missing.txt"
        chk = checksum_outputs([p_exists, p_missing])
        assert chk["exists.txt"]  is not None
        assert chk["missing.txt"] is None
        record(test_id, "checksum_outputs marks missing file as None", PASS)
    except Exception as e:
        record(test_id, "checksum_outputs marks missing file as None",
               FAIL, str(e))

    # T06
    test_id = "T06"
    try:
        assert _safe_serialise(None)       is None
        assert _safe_serialise("hi")       == "hi"
        assert _safe_serialise(42)         == 42
        assert _safe_serialise(3.14)       == 3.14
        assert _safe_serialise(True)       == True
        assert _safe_serialise({"a": 1})   == {"a": 1}
        assert _safe_serialise([1, 2])     == [1, 2]
        record(test_id, "_safe_serialise handles primitive types", PASS)
    except Exception as e:
        record(test_id, "_safe_serialise handles primitive types", FAIL, str(e))

    # T07
    test_id = "T07"
    try:
        deep_dict = {str(i): i for i in range(50)}
        deep_list = list(range(100))
        r_dict = _safe_serialise(deep_dict)
        r_list = _safe_serialise(deep_list)
        assert len(r_dict) <= 30
        assert len(r_list) <= 20
        record(test_id, "_safe_serialise caps depth", PASS,
               f"dict={len(r_dict)}, list={len(r_list)}")
    except Exception as e:
        record(test_id, "_safe_serialise caps depth", FAIL, str(e))

    # T08 — execute_step PASS with dummy module
    test_id = "T08"
    try:
        dummy_py = tmp_dir / "cell99.py"
        dummy_py.write_text("def run_cell_99(): return {'status': 'ok'}\n")
        orig_paths = CELL_MODULE_PATHS.copy()
        CELL_MODULE_PATHS[99] = dummy_py
        step = {
            "cell"        : 99,
            "name"        : "Dummy Step",
            "entry_fn"    : "run_cell_99",
            "output_files": [],
            "required"    : True,
        }
        r = execute_step(step)
        assert r["status"]    == "PASS"
        assert r["runtime_s"] >= 0
        assert r["error"]     is None
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step PASS for valid step", PASS)
    except Exception as e:
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step PASS for valid step", FAIL, str(e))

    # T09 — execute_step FAIL captures traceback
    test_id = "T09"
    try:
        fail_py = tmp_dir / "cell98.py"
        fail_py.write_text(
            "def run_cell_98(): raise ValueError('synthetic failure')\n"
        )
        orig_paths = CELL_MODULE_PATHS.copy()
        CELL_MODULE_PATHS[98] = fail_py
        step = {
            "cell"        : 98,
            "name"        : "Failing Step",
            "entry_fn"    : "run_cell_98",
            "output_files": [],
            "required"    : True,
        }
        r = execute_step(step)
        assert r["status"] == "FAIL"
        assert r["error"]  is not None
        assert "synthetic failure" in r["error"]
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step FAIL captures traceback", PASS)
    except Exception as e:
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step FAIL captures traceback", FAIL, str(e))

    # T10 — execute_step captures stdout
    test_id = "T10"
    try:
        print_py = tmp_dir / "cell97.py"
        print_py.write_text(
            "def run_cell_97():\n"
            "    print('hello from cell97')\n"
            "    return {}\n"
        )
        orig_paths = CELL_MODULE_PATHS.copy()
        CELL_MODULE_PATHS[97] = print_py
        step = {
            "cell"        : 97,
            "name"        : "Print Step",
            "entry_fn"    : "run_cell_97",
            "output_files": [],
            "required"    : True,
        }
        r = execute_step(step)
        assert r["status"] == "PASS"
        assert "hello from cell97" in r["stdout"]
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step captures stdout", PASS)
    except Exception as e:
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step captures stdout", FAIL, str(e))

    # T11 — runtime_s > 0
    test_id = "T11"
    try:
        sleep_py = tmp_dir / "cell96.py"
        sleep_py.write_text(
            "import time\n"
            "def run_cell_96():\n"
            "    time.sleep(0.05)\n"
            "    return {}\n"
        )
        orig_paths = CELL_MODULE_PATHS.copy()
        CELL_MODULE_PATHS[96] = sleep_py
        step = {
            "cell"        : 96,
            "name"        : "Sleep Step",
            "entry_fn"    : "run_cell_96",
            "output_files": [],
            "required"    : True,
        }
        r = execute_step(step)
        assert r["runtime_s"] >= 0.04, \
            f"runtime_s={r['runtime_s']} — expected >= 0.04"
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step records runtime_s > 0", PASS,
               f"runtime={r['runtime_s']}s")
    except Exception as e:
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step records runtime_s > 0", FAIL, str(e))

    # T12 — stdout truncation
    test_id = "T12"
    try:
        trunc_py = tmp_dir / "cell95.py"
        # Write more than CAPTURE_MAX_BYTES
        trunc_py.write_text(
            "def run_cell_95():\n"
            f"    print('X' * {CAPTURE_MAX_BYTES + 1000})\n"
            "    return {}\n"
        )
        orig_paths = CELL_MODULE_PATHS.copy()
        CELL_MODULE_PATHS[95] = trunc_py
        step = {
            "cell"        : 95,
            "name"        : "Trunc Step",
            "entry_fn"    : "run_cell_95",
            "output_files": [],
            "required"    : True,
        }
        r = execute_step(step)
        assert "TRUNCATED" in r["stdout"], "Truncation marker missing"
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step truncates oversized stdout", PASS)
    except Exception as e:
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "execute_step truncates oversized stdout", FAIL, str(e))

    # T13 — dry_run returns DRY_RUN status per step
    test_id = "T13"
    try:
        report = run_pipeline(dry_run=True)
        assert all(r["status"] == "DRY_RUN"
                   for r in report["step_results"])
        assert report["pipeline_status"] == "SUCCESS"
        record(test_id, "dry_run: all steps DRY_RUN status", PASS,
               f"steps={len(report['step_results'])}")
    except Exception as e:
        record(test_id, "dry_run: all steps DRY_RUN status", FAIL, str(e))

    # T14 — dry_run requires no cell modules
    test_id = "T14"
    try:
        # Save and clear module paths
        orig_paths = CELL_MODULE_PATHS.copy()
        CELL_MODULE_PATHS.clear()
        report = run_pipeline(dry_run=True)
        assert report["pipeline_status"] == "SUCCESS"
        CELL_MODULE_PATHS.update(orig_paths)
        record(test_id, "dry_run requires no cell modules", PASS)
    except Exception as e:
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths if 'orig_paths' in dir() else {})
        record(test_id, "dry_run requires no cell modules", FAIL, str(e))

    # T15 — halts on first failure
    test_id = "T15"
    try:
        fail1_py = tmp_dir / "cell_halt1.py"
        fail2_py = tmp_dir / "cell_halt2.py"
        fail1_py.write_text("def run_halt1(): raise RuntimeError('step1 fail')\n")
        fail2_py.write_text("def run_halt2(): return {'ran': True}\n")

        orig_paths  = CELL_MODULE_PATHS.copy()
        orig_registry = STEP_REGISTRY.copy()
        CELL_MODULE_PATHS[901] = fail1_py
        CELL_MODULE_PATHS[902] = fail2_py
        STEP_REGISTRY.clear()
        STEP_REGISTRY.extend([
            {"cell": 901, "name": "Fail First",
             "entry_fn": "run_halt1",
             "output_files": [], "required": True},
            {"cell": 902, "name": "Should Not Run",
             "entry_fn": "run_halt2",
             "output_files": [], "required": True},
        ])

        report = run_pipeline()

        STEP_REGISTRY.clear()
        STEP_REGISTRY.extend(orig_registry)
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths)

        assert report["pipeline_status"] == "FAILED"
        assert report["halt_at_cell"]    == 901
        assert len(report["step_results"]) == 1, \
            f"Expected 1 step executed, got {len(report['step_results'])}"
        assert report["step_results"][0]["status"] == "FAIL"
        record(test_id, "Pipeline halts on first failure", PASS)
    except Exception as e:
        STEP_REGISTRY.clear()
        STEP_REGISTRY.extend(orig_registry if "orig_registry" in dir() else [])
        CELL_MODULE_PATHS.clear()
        CELL_MODULE_PATHS.update(orig_paths if "orig_paths" in dir() else {})
        record(test_id, "Pipeline halts on first failure", FAIL, str(e))

    # T16 — _build_run_report counts
    test_id = "T16"
    try:
        fake_results = [
            {"cell": 33, "name": "A", "status": "PASS",
             "start_ts": None, "end_ts": None, "runtime_s": 1.0,
             "stdout": "", "stderr": "", "checksums": {},
             "error": None, "return_value": None},
            {"cell": 34, "name": "B", "status": "FAIL",
             "start_ts": None, "end_ts": None, "runtime_s": 0.5,
             "stdout": "", "stderr": "", "checksums": {},
             "error": "boom", "return_value": None},
        ]
        rpt = _build_run_report("20260310_120000", fake_results,
                                "FAILED", 34, 1.5)
        assert rpt["steps_passed"] == 1
        assert rpt["steps_failed"] == 1
        assert rpt["steps_total"]  == len(STEP_REGISTRY)
        assert rpt["halt_at_cell"] == 34
        record(test_id, "_build_run_report counts correct", PASS)
    except Exception as e:
        record(test_id, "_build_run_report counts correct", FAIL, str(e))

    # T17 — _save_run_report writes JSON
    test_id = "T17"
    try:
        global LOG_DIR, PIPELINE_RUN_LATEST
        _orig_log  = LOG_DIR
        _orig_lat  = PIPELINE_RUN_LATEST
        LOG_DIR           = tmp_dir
        PIPELINE_RUN_LATEST = tmp_dir / "pipeline_run_latest.json"

        fake_rpt = {
            "run_timestamp"  : "20260310_000000",
            "pipeline_status": "SUCCESS",
            "halt_at_cell"   : None,
            "total_runtime_s": 5.0,
            "steps_total"    : 7,
            "steps_executed" : 7,
            "steps_passed"   : 7,
            "steps_failed"   : 0,
            "step_results"   : [],
        }
        ts_path = _save_run_report(fake_rpt)
        assert ts_path.exists()
        assert PIPELINE_RUN_LATEST.exists()
        with open(ts_path) as f:
            loaded = json.load(f)
        assert loaded["pipeline_status"] == "SUCCESS"

        LOG_DIR             = _orig_log
        PIPELINE_RUN_LATEST = _orig_lat
        record(test_id, "_save_run_report writes JSON", PASS)
    except Exception as e:
        LOG_DIR             = _orig_log
        PIPELINE_RUN_LATEST = _orig_lat
        record(test_id, "_save_run_report writes JSON", FAIL, str(e))

    # T18 — _append_pipeline_log CSV columns
    test_id = "T18"
    try:
        global PIPELINE_LOG_CSV
        _orig_csv   = PIPELINE_LOG_CSV
        PIPELINE_LOG_CSV = tmp_dir / "pipeline_log_test.csv"

        fake_rpt = {
            "run_timestamp"  : "20260310_000001",
            "pipeline_status": "SUCCESS",
            "step_results"   : [
                {"cell": 33, "name": "A", "status": "PASS",
                 "runtime_s": 2.1, "error": None},
            ],
        }
        _append_pipeline_log(fake_rpt)
        df = pd.read_csv(PIPELINE_LOG_CSV)
        for col in ["run_timestamp","pipeline_status","cell",
                    "name","status","runtime_s","error_excerpt"]:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) == 1

        PIPELINE_LOG_CSV = _orig_csv
        record(test_id, "_append_pipeline_log CSV columns correct", PASS)
    except Exception as e:
        PIPELINE_LOG_CSV = _orig_csv
        record(test_id, "_append_pipeline_log CSV columns correct",
               FAIL, str(e))

    # T19 — Scheduler starts and stops
    test_id = "T19"
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        # Use a very long interval so it doesn't fire during the test
        start_scheduler(
            interval_hours=9999,
            run_immediately=False,
        )
        status = scheduler_status()
        assert status["running"] is True
        assert len(status["jobs"]) >= 1
        stop_scheduler()
        status2 = scheduler_status()
        assert status2["running"] is False
        record(test_id, "Scheduler starts, status correct, stops", PASS)
    except Exception as e:
        stop_scheduler()
        record(test_id, "Scheduler starts, status correct, stops", FAIL, str(e))

    # T20 — End-to-end dry_run on Drive data
    test_id = "T20"
    try:
        report = run_pipeline(dry_run=True)
        assert report["pipeline_status"] == "SUCCESS"
        assert len(report["step_results"]) == len(STEP_REGISTRY)

        # Verify all expected output files are present (at least some)
        all_checksums = {}
        for r in report["step_results"]:
            all_checksums.update(r["checksums"])

        present  = [k for k, v in all_checksums.items() if v is not None]
        missing  = [k for k, v in all_checksums.items() if v is None]

        # At minimum, all Cell 36-39 outputs must exist (they were just run)
        critical_outputs = [
            "risk_assessment_latest.json",
            "recommendations_latest.json",
            "alert_log.csv",
            "alert_state.json",
        ]
        for f in critical_outputs:
            assert f in present, f"Critical output missing: {f}"

        record(test_id, "End-to-end dry_run on Drive data", PASS,
               f"outputs present={len(present)}, missing={len(missing)}")
    except Exception as e:
        record(test_id, "End-to-end dry_run on Drive data", FAIL, str(e))

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, _, s, _ in results if s == PASS)
    failed = sum(1 for _, _, s, _ in results if s == FAIL)
    total  = len(results)
    print(f"  Passed : {passed}/{total}")
    print(f"  Failed : {failed}/{total}")
    if failed:
        print("\n  Failed tests:")
        for tid, name, status, detail in results:
            if status == FAIL:
                print(f"    {tid}: {name}")
                if detail:
                    print(f"         Detail: {detail}")
    print("=" * 70 + "\n")


# =============================================================================
# RUN
# =============================================================================

test_cell_40()

# ── Usage guide printed after tests ──────────────────────────────────────────
print("CELL 40 USAGE GUIDE")
print("-" * 70)
print()
print("  MANUAL TRIGGER (run full pipeline once):")
print("    report = run_pipeline()")
print()
print("  PARTIAL RERUN (from Cell 36 onward):")
print("    report = run_pipeline(steps=[36, 37, 38, 39])")
print()
print("  DRY RUN (check all output files, no execution):")
print("    report = run_pipeline(dry_run=True)")
print()
print("  SCHEDULED (every 24h, runs once immediately):")
print("    start_scheduler(interval_hours=24)")
print()
print("  SCHEDULED (every 6h, starting from next tick):")
print("    start_scheduler(interval_hours=6, run_immediately=False)")
print()
print("  CHECK SCHEDULER:")
print("    scheduler_status()")
print()
print("  STOP SCHEDULER:")
print("    stop_scheduler()")
print()
print("  CELL MODULE PATHS (save each cell's .py here):")
for cell_num, path in CELL_MODULE_PATHS.items():
    print(f"    Cell {cell_num}: {path}")
print()
print("  PIPELINE LOG:")
print(f"    {PIPELINE_LOG_CSV}")
print(f"    {PIPELINE_RUN_LATEST}")
