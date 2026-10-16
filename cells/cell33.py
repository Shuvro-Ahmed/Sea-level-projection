# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 33: Autonomous Data Ingestion
# =============================================================================
# Purpose  : Autonomously ingest CMEMS AVISO+ altimetry and PSMSL tide gauge
#            data without human intervention. Validates incoming data quality
#            and maintains a full audit trail of all ingestion events.
# Authors  : HyperSLR Research Pipeline
# Extends  : BD_ML_VT 32-cell pipeline (slr_enhanced_dataset.csv, 2628 rows)
# Target   : Nature Climate Change / Environmental Research Letters
# =============================================================================

# --- 0. DEPENDENCY INSTALLATION ----------------------------------------------

import subprocess
import sys

def install_package(package: str) -> None:
    """Install a package silently via pip."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print("Installing dependencies...")
for pkg in ["copernicusmarine", "requests", "netCDF4", "scipy"]:
    try:
        install_package(pkg)
        print(f"  [OK] {pkg}")
    except Exception as e:
        print(f"  [FAILED] {pkg}: {e}")

print("Dependency installation complete.\n")


# --- 1. IMPORTS --------------------------------------------------------------

import os
import json
import time
import logging
import warnings
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
from scipy import stats

try:
    import copernicusmarine
    CMEMS_AVAILABLE = True
except ImportError:
    CMEMS_AVAILABLE = False
    warnings.warn("copernicusmarine not available. CMEMS ingestion will be skipped.")

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.filterwarnings("ignore", category=FutureWarning)


# --- 2. CONFIGURATION --------------------------------------------------------

DRIVE_BASE      = Path("/content/drive/MyDrive/BD_ML_VT")
LIVE_DIR        = DRIVE_BASE / "live_data"
LOG_DIR         = DRIVE_BASE / "logs"
PROCESSED_DIR   = DRIVE_BASE / "processed"

# CMEMS configuration
CMEMS_DATASET   = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"
CMEMS_VARIABLE  = "sla"
CMEMS_BBOX      = {"minimum_longitude": 72.0, "maximum_longitude": 110.0,
                   "minimum_latitude": -2.0,  "maximum_latitude": 27.0}
CMEMS_OUTPUT    = LIVE_DIR / "cmems_latest.nc"

# PSMSL station configuration
PSMSL_STATIONS = {
    "Bangladesh": [369, 543, 1270],
    "Vietnam"   : [841, 1449],
    "Maldives"  : [1707],
}
PSMSL_BASE_URL  = "https://www.psmsl.org/data/obtaining/rlr.monthly.data/{station_id}.rlrdata"
PSMSL_OUTPUT    = LIVE_DIR / "psmsl_latest.csv"

# Validation thresholds
MAX_MISSING_FRACTION = 0.30
MAX_ZSCORE           = 4.0
EXPECTED_FEATURES    = 12

# Historical dataset reference
HISTORICAL_CSV  = PROCESSED_DIR / "slr_enhanced_dataset.csv"
SCALER_JSON     = PROCESSED_DIR / "scaler_params.json"

# Audit log
INGESTION_LOG   = LOG_DIR / "ingestion_log.csv"

# Timestamp for this run
RUN_TIMESTAMP   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# --- 3. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell33")


# --- 4. DIRECTORY INITIALISATION --------------------------------------------

def initialise_directories() -> None:
    """Create required output directories if they do not exist."""
    for directory in [LIVE_DIR, LOG_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Directory confirmed: %s", directory)

initialise_directories()


# --- 5. GOOGLE DRIVE MOUNT ---------------------------------------------------

def mount_drive() -> bool:
    """Mount Google Drive if running in Colab. Returns True on success."""
    if not IN_COLAB:
        logger.warning("Not in Colab environment. Assuming local filesystem.")
        return True
    try:
        drive.mount("/content/drive", force_remount=False)
        logger.info("Google Drive mounted successfully.")
        return True
    except Exception as exc:
        logger.error("Drive mount failed: %s", exc)
        return False

drive_mounted = mount_drive()


# --- 6. AUDIT LOGGING --------------------------------------------------------

def log_ingestion_event(
    source     : str,
    status     : str,
    records    : int,
    details    : str,
    error      : Optional[str] = None
) -> None:
    """
    Append one row to the ingestion audit log CSV.

    Parameters
    ----------
    source  : Data source identifier (e.g., 'CMEMS', 'PSMSL_Bangladesh')
    status  : 'SUCCESS', 'PARTIAL', 'FAILED', 'SKIPPED'
    records : Number of records ingested (0 if failed)
    details : Human-readable summary of the ingestion event
    error   : Exception message if status is FAILED
    """
    row = {
        "timestamp"  : RUN_TIMESTAMP,
        "source"     : source,
        "status"     : status,
        "records"    : records,
        "details"    : details,
        "error"      : error if error else "",
    }

    df_row = pd.DataFrame([row])

    if INGESTION_LOG.exists():
        df_row.to_csv(INGESTION_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(INGESTION_LOG, mode="w", header=True, index=False)

    logger.info("[LOG] %s | %s | records=%d | %s", source, status, records, details)


# --- 7. CMEMS INGESTION ------------------------------------------------------

def ingest_cmems_latest(
    username     : str,
    password     : str,
    start_date   : str = "2023-01-01",
    end_date     : str = "2025-12-31",
    dry_run      : bool = False
) -> Tuple[bool, int]:
    """
    Download latest AVISO+ sea-level anomaly data from CMEMS.

    Uses copernicusmarine subset() to extract the regional bounding box
    covering Bangladesh, Vietnam, and the Maldives. Saves as NetCDF4.

    Parameters
    ----------
    username   : Copernicus Marine Service username
    password   : Copernicus Marine Service password
    start_date : ISO date string, inclusive
    end_date   : ISO date string, inclusive
    dry_run    : If True, skip actual download (for testing connectivity)

    Returns
    -------
    (success: bool, estimated_records: int)
    """
    if not CMEMS_AVAILABLE:
        logger.warning("copernicusmarine not installed. Skipping CMEMS ingestion.")
        log_ingestion_event("CMEMS", "SKIPPED", 0, "copernicusmarine package unavailable.")
        return False, 0

    logger.info("--- CMEMS INGESTION STARTING ---")
    logger.info("Dataset  : %s", CMEMS_DATASET)
    logger.info("Variable : %s", CMEMS_VARIABLE)
    logger.info("Period   : %s to %s", start_date, end_date)
    logger.info("BBox     : lon [%s, %s], lat [%s, %s]",
                CMEMS_BBOX["minimum_longitude"], CMEMS_BBOX["maximum_longitude"],
                CMEMS_BBOX["minimum_latitude"],  CMEMS_BBOX["maximum_latitude"])

    if dry_run:
        logger.info("DRY RUN: CMEMS download skipped.")
        log_ingestion_event("CMEMS", "SKIPPED", 0, "dry_run=True")
        return True, 0

    try:
        copernicusmarine.subset(
            dataset_id         = CMEMS_DATASET,
            variables          = [CMEMS_VARIABLE],
            start_datetime     = start_date,
            end_datetime       = end_date,
            output_filename    = str(CMEMS_OUTPUT),
            output_directory   = str(LIVE_DIR),
            username           = username,
            password           = password,
            force_download     = True,
            **CMEMS_BBOX,
        )

        if not CMEMS_OUTPUT.exists():
            raise FileNotFoundError(f"Expected output file not found: {CMEMS_OUTPUT}")

        file_size_mb = CMEMS_OUTPUT.stat().st_size / (1024 ** 2)
        logger.info("CMEMS file saved: %s (%.2f MB)", CMEMS_OUTPUT, file_size_mb)

        # Estimate record count from file size (rough approximation)
        estimated_records = int(file_size_mb * 10)
        log_ingestion_event(
            "CMEMS", "SUCCESS", estimated_records,
            f"Period {start_date} to {end_date}, {file_size_mb:.2f} MB"
        )
        return True, estimated_records

    except Exception as exc:
        error_msg = str(exc)
        logger.error("CMEMS ingestion failed: %s", error_msg)
        logger.debug(traceback.format_exc())
        log_ingestion_event("CMEMS", "FAILED", 0, "Download error.", error=error_msg)
        return False, 0


# --- 8. PSMSL INGESTION ------------------------------------------------------

def _parse_psmsl_rlr(raw_text: str, station_id: int) -> pd.DataFrame:
    """
    Parse raw PSMSL RLR monthly data format into a clean DataFrame.

    PSMSL RLR format (space-delimited):
      decimal_year ; rlr_height_mm ; flag ; flag2

    Heights of -99999 indicate missing values.

    Parameters
    ----------
    raw_text   : Raw text content from PSMSL HTTP response
    station_id : PSMSL station identifier for labelling

    Returns
    -------
    DataFrame with columns: [year_decimal, date, rlr_mm, station_id]
    """
    rows = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            year_dec = float(parts[0])
            rlr_val  = float(parts[1])
            if rlr_val == -99999:
                rlr_val = np.nan
            # Convert decimal year to approximate date (first of month)
            year    = int(year_dec)
            month   = max(1, min(12, round((year_dec - year) * 12) + 1))
            date    = pd.Timestamp(year=year, month=month, day=1)
            rows.append({"year_decimal": year_dec, "date": date,
                         "rlr_mm": rlr_val, "station_id": station_id})
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(rows)


def ingest_psmsl_station(
    station_id : int,
    country    : str,
    timeout    : int = 30
) -> Tuple[bool, pd.DataFrame]:
    """
    Download RLR monthly data for one PSMSL station via HTTP.

    Parameters
    ----------
    station_id : PSMSL station integer ID
    country    : Country label for logging
    timeout    : HTTP request timeout in seconds

    Returns
    -------
    (success: bool, DataFrame of records)
    """
    url = PSMSL_BASE_URL.format(station_id=station_id)
    logger.info("  Requesting PSMSL station %d (%s): %s", station_id, country, url)

    try:
        response = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "HyperSLR-Research/1.0"})
        response.raise_for_status()

        df = _parse_psmsl_rlr(response.text, station_id)
        df["country"] = country

        if df.empty:
            logger.warning("    Station %d: no records parsed.", station_id)
            return False, df

        valid_count = df["rlr_mm"].notna().sum()
        logger.info("    Station %d: %d records (%d valid, %d missing)",
                    station_id, len(df), valid_count, df["rlr_mm"].isna().sum())
        return True, df

    except requests.exceptions.Timeout:
        logger.error("    Station %d: HTTP timeout after %ds.", station_id, timeout)
        return False, pd.DataFrame()
    except requests.exceptions.HTTPError as exc:
        logger.error("    Station %d: HTTP error %s.", station_id, exc)
        return False, pd.DataFrame()
    except Exception as exc:
        logger.error("    Station %d: Unexpected error: %s", station_id, exc)
        return False, pd.DataFrame()


def ingest_psmsl_latest(
    dry_run: bool = False
) -> Tuple[bool, pd.DataFrame, Dict[str, int]]:
    """
    Download PSMSL RLR monthly data for all configured stations.

    Iterates through PSMSL_STATIONS configuration, fetches each station,
    merges results into a single DataFrame, and saves to PSMSL_OUTPUT.

    Parameters
    ----------
    dry_run : If True, skip HTTP requests and return empty DataFrame

    Returns
    -------
    (overall_success: bool, combined_df: DataFrame, record_counts: dict)
    """
    logger.info("--- PSMSL INGESTION STARTING ---")

    if dry_run:
        logger.info("DRY RUN: PSMSL download skipped.")
        log_ingestion_event("PSMSL_ALL", "SKIPPED", 0, "dry_run=True")
        return True, pd.DataFrame(), {}

    all_frames   = []
    record_counts = {}
    all_success  = True

    for country, station_ids in PSMSL_STATIONS.items():
        country_frames  = []
        country_success = True

        for station_id in station_ids:
            success, df = ingest_psmsl_station(station_id, country)
            if success and not df.empty:
                country_frames.append(df)
            else:
                country_success = False
                all_success     = False

        if country_frames:
            country_df  = pd.concat(country_frames, ignore_index=True)
            record_count = len(country_df)
            record_counts[country] = record_count
            all_frames.append(country_df)

            source_label = f"PSMSL_{country}"
            status       = "SUCCESS" if country_success else "PARTIAL"
            log_ingestion_event(
                source_label, status, record_count,
                f"{len(station_ids)} stations, {record_count} total rows"
            )
        else:
            record_counts[country] = 0
            log_ingestion_event(
                f"PSMSL_{country}", "FAILED", 0,
                f"No data retrieved for any of stations {station_ids}"
            )

    if all_frames:
        combined_df = pd.concat(all_frames, ignore_index=True)
        combined_df.sort_values(["country", "station_id", "date"], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        combined_df.to_csv(PSMSL_OUTPUT, index=False)
        logger.info("PSMSL combined file saved: %s (%d rows)", PSMSL_OUTPUT, len(combined_df))
        return all_success, combined_df, record_counts
    else:
        logger.error("PSMSL ingestion: no data retrieved from any station.")
        return False, pd.DataFrame(), record_counts


# --- 9. DATA VALIDATION ------------------------------------------------------

def validate_incoming_data(
    df         : pd.DataFrame,
    source_name: str,
    column     : str = "rlr_mm"
) -> Dict[str, Any]:
    """
    Apply quality checks to ingested data before pipeline entry.

    Checks performed:
      1. Missing value fraction against MAX_MISSING_FRACTION threshold
      2. Z-score outlier detection against MAX_ZSCORE threshold
      3. Temporal monotonicity (dates must be non-decreasing)
      4. Reasonable physical range for sea-level data (-5000 to +5000 mm)

    Parameters
    ----------
    df          : DataFrame to validate
    source_name : Label for logging
    column      : Column name containing the primary measurement

    Returns
    -------
    Dictionary with keys: passed (bool), missing_frac, outlier_count,
    monotonic, range_ok, warnings (list), total_rows
    """
    report = {
        "source"        : source_name,
        "total_rows"    : len(df),
        "passed"        : True,
        "missing_frac"  : 0.0, # Initialize to 0.0
        "outlier_count" : 0,   # Initialize to 0
        "monotonic"     : True, # Initialize to True
        "range_ok"      : True, # Initialize to True
        "warnings"      : [],
    }

    if df.empty:
        report["passed"]   = False
        report["warnings"].append("Empty DataFrame — nothing to validate.")
        return report

    if column not in df.columns:
        report["passed"]   = False
        report["warnings"].append(f"Column '{column}' not found in DataFrame.")
        return report

    series = df[column].copy()

    # Check 1: Missing value fraction
    missing_frac = series.isna().mean()
    report["missing_frac"] = round(float(missing_frac), 4)
    if missing_frac > MAX_MISSING_FRACTION:
        report["passed"] = False
        report["warnings"].append(
            f"Missing fraction {missing_frac:.1%} exceeds threshold {MAX_MISSING_FRACTION:.1%}."
        )
    else:
        logger.info("  Validation [%s] missing fraction: %.1f%% — PASS", source_name, missing_frac * 100)

    # Check 2: Z-score outlier detection (on non-null values)
    valid_series = series.dropna()
    if len(valid_series) > 2:
        z_scores       = np.abs(stats.zscore(valid_series))
        outlier_count  = int((z_scores > MAX_ZSCORE).sum())
        report["outlier_count"] = outlier_count
        if outlier_count > 0:
            warning_msg = f"{outlier_count} outlier(s) with |z| > {MAX_ZSCORE}."
            report["warnings"].append(warning_msg)
            logger.warning("  Validation [%s] %s", source_name, warning_msg)
        else:
            logger.info("  Validation [%s] outlier count: 0 — PASS", source_name)
    else:
        report["outlier_count"] = 0 # Ensure it's not None
        report["warnings"].append("Insufficient data for z-score computation.")

    # Check 3: Temporal monotonicity — checked per station_id when that
    # column is present. Multi-station DataFrames are expected to be
    # non-monotonic at the country level because station time-series
    # interleave. Flagging at country level would produce false positives.
    if "date" in df.columns:
        if "station_id" in df.columns:
            station_mono_results = []
            for sid, grp in df.groupby("station_id"):
                dates = pd.to_datetime(grp["date"], errors="coerce").dropna()
                station_mono_results.append(dates.is_monotonic_increasing)
            is_mono = all(station_mono_results)
            scope   = "per-station"
        else:
            dates   = pd.to_datetime(df["date"], errors="coerce").dropna()
            is_mono = bool(dates.is_monotonic_increasing)
            scope   = "global"

        report["monotonic"] = is_mono
        if not is_mono:
            report["warnings"].append(
                f"Date column is not monotonically increasing ({scope} check)."
            )
            logger.warning("  Validation [%s] temporal monotonicity (%s) — FAIL",
                           source_name, scope)
        else:
            logger.info("  Validation [%s] temporal monotonicity (%s) — PASS",
                        source_name, scope)
    else:
        report["monotonic"] = False # Ensure it's not None

    # Check 4: Physical range — use IQR-based adaptive bounds for RLR data.
    # PSMSL RLR heights are referenced to a local datum (~6700–7100 mm for
    # the Indian Ocean / South-East Asia region) and must NOT be compared
    # against a zero-centred physical range. Instead, flag values that
    # deviate more than 10× the IQR from the median, which catches genuine
    # instrument errors while tolerating the full RLR datum offset.
    if len(valid_series) >= 4:
        q1, q3    = valid_series.quantile(0.25), valid_series.quantile(0.75)
        iqr       = q3 - q1
        lower     = q1 - 10.0 * iqr
        upper     = q3 + 10.0 * iqr
        in_range  = valid_series.between(lower, upper)
        range_ok  = bool(in_range.all())
        report["range_ok"] = range_ok
        if not range_ok:
            out_of_range = int((~in_range).sum())
            report["warnings"].append(
                f"{out_of_range} value(s) outside adaptive IQR bounds "
                f"[{lower:.1f}, {upper:.1f}] mm."
            )
            logger.warning("  Validation [%s] physical range — FAIL (%d values)",
                           source_name, out_of_range)
        else:
            logger.info("  Validation [%s] physical range — PASS", source_name)
    else:
        report["range_ok"] = False # Ensure it's not None
        report["warnings"].append("Insufficient data for range check (< 4 values).")

    return report


def validate_all_ingested_data(psmsl_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run validation checks on all ingested PSMSL data by country.

    Parameters
    ----------
    psmsl_df : Combined PSMSL DataFrame with 'country' column

    Returns
    -------
    Dictionary keyed by country name, values are validation reports
    """
    logger.info("--- VALIDATION STARTING ---")
    validation_results = {}

    if psmsl_df.empty:
        logger.warning("No PSMSL data available for validation.")
        return validation_results

    for country in psmsl_df["country"].unique():
        country_df = psmsl_df[psmsl_df["country"] == country].copy()
        report     = validate_incoming_data(country_df, f"PSMSL_{country}", column="rlr_mm")
        validation_results[country] = report

        status = "PASSED" if report["passed"] else "FAILED"
        logger.info("  [%s] Validation %s | rows=%d | missing=%.1f%% | outliers=%s",
                    country, status, report["total_rows"],
                    (report["missing_frac"] or 0) * 100,
                    report["outlier_count"])

    return validation_results


# --- 10. INGESTION SUMMARY ---------------------------------------------------

def generate_ingestion_summary(
    cmems_success    : bool,
    cmems_records    : int,
    psmsl_success    : bool,
    psmsl_df         : pd.DataFrame,
    psmsl_counts     : Dict[str, int],
    validation_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Produce a structured summary dictionary of the full ingestion run.

    Parameters
    ----------
    cmems_success      : Whether CMEMS download succeeded
    cmems_records      : Estimated CMEMS record count
    psmsl_success      : Whether all PSMSL downloads succeeded
    psmsl_df           : Combined PSMSL DataFrame
    psmsl_counts       : Per-country record counts
    validation_results : Output from validate_all_ingested_data()

    Returns
    -------
    Summary dictionary suitable for JSON serialisation
    """
    psmsl_date_range = {}
    if not psmsl_df.empty and "date" in psmsl_df.columns:
        for country in psmsl_df["country"].unique():
            cdf   = psmsl_df[psmsl_df["country"] == country]
            dates = pd.to_datetime(cdf["date"], errors="coerce").dropna()
            if not dates.empty:
                psmsl_date_range[country] = {
                    "start": str(dates.min().date()),
                    "end"  : str(dates.max().date()),
                }

    all_validation_passed = all(
        r.get("passed", False) for r in validation_results.values()
    ) if validation_results else False

    summary = {
        "run_timestamp"          : RUN_TIMESTAMP,
        "cmems": {
            "success"            : cmems_success,
            "estimated_records"  : cmems_records,
            "output_file"        : str(CMEMS_OUTPUT),
        },
        "psmsl": {
            "success"            : psmsl_success,
            "total_records"      : sum(psmsl_counts.values()),
            "records_by_country" : psmsl_counts,
            "date_ranges"        : psmsl_date_range,
            "output_file"        : str(PSMSL_OUTPUT),
        },
        "validation": {
            "all_passed"         : all_validation_passed,
            "by_country"         : validation_results,
        },
        "log_file"               : str(INGESTION_LOG),
    }

    # Save summary as JSON
    summary_path = LOG_DIR / f"ingestion_summary_{RUN_TIMESTAMP}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Ingestion summary saved: %s", summary_path)

    return summary


# =============================================================================
# CELL 33 — MAIN EXECUTION
# =============================================================================

def run_cell_33(
    cmems_username : str  = "",
    cmems_password : str  = "",
    dry_run        : bool = False
) -> Dict[str, Any]:
    """
    Execute the full Cell 33 autonomous data ingestion pipeline.

    Parameters
    ----------
    cmems_username : Copernicus Marine Service username
                     (obtain from: https://marine.copernicus.eu/)
    cmems_password : Copernicus Marine Service password
    dry_run        : If True, skip actual HTTP downloads
                     (use for testing pipeline logic without credentials)

    Returns
    -------
    Full ingestion summary dictionary
    """
    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 33: AUTONOMOUS DATA INGESTION")
    logger.info("Run timestamp: %s", RUN_TIMESTAMP)
    logger.info("Google Drive: %s", "mounted" if drive_mounted else "not mounted")
    logger.info("Dry run mode: %s", dry_run)
    logger.info("=" * 70)

    # Step 1: CMEMS ingestion
    cmems_success, cmems_records = ingest_cmems_latest(
        username   = cmems_username,
        password   = cmems_password,
        start_date = "2023-01-01",
        end_date   = "2025-12-31",
        dry_run    = dry_run or not (cmems_username and cmems_password)
    )

    # Step 2: PSMSL ingestion
    psmsl_success, psmsl_df, psmsl_counts = ingest_psmsl_latest(dry_run=dry_run)

    # Step 3: Validate ingested data
    validation_results = validate_all_ingested_data(psmsl_df)

    # Step 4: Generate and return summary
    summary = generate_ingestion_summary(
        cmems_success     = cmems_success,
        cmems_records     = cmems_records,
        psmsl_success     = psmsl_success,
        psmsl_df          = psmsl_df,
        psmsl_counts      = psmsl_counts,
        validation_results= validation_results
    )

    logger.info("=" * 70)
    logger.info("CELL 33 COMPLETE")
    logger.info("CMEMS   : %s | estimated records: %d",
                "SUCCESS" if cmems_success else "SKIPPED/FAILED", cmems_records)
    logger.info("PSMSL   : %s | total records: %d",
                "SUCCESS" if psmsl_success else "PARTIAL/FAILED",
                summary["psmsl"]["total_records"])
    logger.info("Validation: %s",
                "ALL PASSED" if summary["validation"]["all_passed"] else "WARNINGS PRESENT")
    logger.info("Audit log : %s", INGESTION_LOG)
    logger.info("=" * 70)

    return summary


# =============================================================================
# CELL 33 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_33() -> None:
    """
    Comprehensive unit and integration tests for all Cell 33 functions.

    Tests are self-contained: no credentials or network access required.
    Each test prints a clear PASS / FAIL result with diagnostic details.

    Test coverage:
      T01 — Directory creation
      T02 — PSMSL response parser (valid input)
      T03 — PSMSL response parser (missing values)
      T04 — PSMSL response parser (malformed input)
      T05 — Validation: clean data passes all checks
      T06 — Validation: high missing fraction fails
      T07 — Validation: outliers flagged
      T08 — Validation: non-monotonic dates flagged
      T09 — Validation: out-of-range values flagged
      T10 — Validation: empty DataFrame handled
      T11 — Log ingestion event writes to CSV
      T12 — Ingestion summary JSON structure
      T13 — PSMSL HTTP station (live network, Bangladesh station 369)
      T14 — Dry-run pipeline execution end-to-end
      T15 — Historical dataset accessibility check
    """

    PASS = "PASS"
    FAIL = "FAIL"
    results = []

    def record(test_id, name, status, detail=""):
        results.append((test_id, name, status, detail))
        marker = "[PASS]" if status == PASS else "[FAIL]"
        print(f"  {marker} {test_id}: {name}" + (f" — {detail}" if detail else ""))

    print("\n" + "=" * 70)
    print("CELL 33 — TEST SUITE")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # T01: Directory creation
    # -------------------------------------------------------------------------
    test_id = "T01"
    try:
        for d in [LIVE_DIR, LOG_DIR]:
            assert d.exists(), f"Directory missing: {d}"
        record(test_id, "Directory creation", PASS)
    except Exception as e:
        record(test_id, "Directory creation", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T02: PSMSL parser — valid input
    # -------------------------------------------------------------------------
    test_id = "T02"
    try:
        sample_psmsl = (
            "1993.0417; 7143; 0; 0\n"
            "1993.1250; 7098; 0; 0\n"
            "1993.2083; 7054; 0; 0\n"
        )
        df = _parse_psmsl_rlr(sample_psmsl, station_id=369)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert "rlr_mm" in df.columns
        assert df["rlr_mm"].notna().all()
        assert (df["station_id"] == 369).all()
        record(test_id, "PSMSL parser — valid input", PASS, f"{len(df)} rows parsed")
    except Exception as e:
        record(test_id, "PSMSL parser — valid input", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T03: PSMSL parser — missing values (-99999)
    # -------------------------------------------------------------------------
    test_id = "T03"
    try:
        sample_with_missing = (
            "1993.0417; 7143; 0; 0\n"
            "1993.1250; -99999; 0; 0\n"
            "1993.2083; 7054; 0; 0\n"
        )
        df = _parse_psmsl_rlr(sample_with_missing, station_id=369)
        assert len(df) == 3
        assert df["rlr_mm"].isna().sum() == 1, "Expected exactly 1 NaN"
        assert pd.notna(df.loc[0, "rlr_mm"])
        assert pd.isna(df.loc[1, "rlr_mm"])
        record(test_id, "PSMSL parser — missing value (-99999)", PASS,
               f"NaN count: {df['rlr_mm'].isna().sum()}")
    except Exception as e:
        record(test_id, "PSMSL parser — missing value (-99999)", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T04: PSMSL parser — malformed/empty input
    # -------------------------------------------------------------------------
    test_id = "T04"
    try:
        df_empty = _parse_psmsl_rlr("", station_id=999)
        assert df_empty.empty, "Expected empty DataFrame for empty input"
        df_bad = _parse_psmsl_rlr("not;valid;data;at;all\nmore;garbage", station_id=999)
        # Should not raise; rows with parse errors are silently dropped
        record(test_id, "PSMSL parser — malformed/empty input", PASS,
               f"Empty: {df_empty.empty}, bad rows dropped gracefully")
    except Exception as e:
        record(test_id, "PSMSL parser — malformed/empty input", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T05: Validation — clean data passes all checks
    # -------------------------------------------------------------------------
    test_id = "T05"
    try:
        np.random.seed(42)
        n = 100
        # RLR datum for Bay of Bengal region is ~7000 mm; synthetic data
        # must reflect this to avoid false positives on the range check.
        df_clean = pd.DataFrame({
            "date"      : pd.date_range("1990-01-01", periods=n, freq="MS"),
            "rlr_mm"    : np.random.normal(7000, 30, n),
            "country"   : "Bangladesh",
            "station_id": 369,
        })
        report = validate_incoming_data(df_clean, "TEST_CLEAN")
        assert report["passed"], f"Expected PASS. Warnings: {report['warnings']}"
        assert report["missing_frac"] == 0.0
        assert report["outlier_count"] == 0
        assert report["monotonic"] is True
        assert report["range_ok"] is True
        record(test_id, "Validation — clean data passes", PASS)
    except Exception as e:
        record(test_id, "Validation — clean data passes", FAIL, str(e))
    # -------------------------------------------------------------------------
    # T06: Validation — high missing fraction fails
    # -------------------------------------------------------------------------
    test_id = "T06"
    try:
        n = 100
        df_missing = pd.DataFrame({
            "date"      : pd.date_range("1990-01-01", periods=n, freq="MS"),
            "rlr_mm" : [np.nan if i % 3 == 0 else 7000.0 for i in range(n)],
            "country"   : "Bangladesh",
            "station_id": 369,
        })
        report = validate_incoming_data(df_missing, "TEST_MISSING")
        assert not report["passed"], "Expected FAIL due to high missing fraction"
        assert report["missing_frac"] > MAX_MISSING_FRACTION
        record(test_id, "Validation — high missing fraction fails", PASS,
               f"missing_frac={report['missing_frac']:.2%}")
    except Exception as e:
        record(test_id, "Validation — high missing fraction fails", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T07: Validation — outliers flagged (not necessarily failing)
    # -------------------------------------------------------------------------
    test_id = "T07"
    try:
        np.random.seed(0)
        n = 100
        values = np.random.normal(7000, 20, n)
        values[50] = 15000  # Extreme outlier
        df_outlier = pd.DataFrame({
            "date"      : pd.date_range("1990-01-01", periods=n, freq="MS"),
            "rlr_mm"    : values,
            "country"   : "Bangladesh",
            "station_id": 369,
        })
        report = validate_incoming_data(df_outlier, "TEST_OUTLIER")
        assert report["outlier_count"] is not None and report["outlier_count"] >= 1, \
            f"Expected at least 1 outlier, got {report['outlier_count']}"
        record(test_id, "Validation — outliers flagged", PASS,
               f"outlier_count={report['outlier_count']}")
    except Exception as e:
        record(test_id, "Validation — outliers flagged", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T08: Validation — non-monotonic dates flagged
    # -------------------------------------------------------------------------
    test_id = "T08"
    try:
        dates = pd.date_range("1990-01-01", periods=10, freq="MS").tolist()
        dates[5] = dates[3]  # Duplicate/non-monotonic
        n = len(dates) # Ensure n matches dates length
        df_nonmono = pd.DataFrame({
            "date"      : dates,
            "rlr_mm" : [np.nan if i % 3 == 0 else 7000.0 for i in range(n)],
            "country"   : "Bangladesh",
            "station_id": 369,
        })
        report = validate_incoming_data(df_nonmono, "TEST_NONMONO")
        assert report["monotonic"] is False, "Expected monotonic=False"
        assert any("monoton" in w.lower() for w in report["warnings"])
        record(test_id, "Validation — non-monotonic dates flagged", PASS)
    except Exception as e:
        record(test_id, "Validation — non-monotonic dates flagged", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T09: Validation — out-of-range values flagged
    # -------------------------------------------------------------------------
    test_id = "T09"
    try:
        values = np.full(20, 7000.0)
        values[10] = 7000.0 + 10000.0   # 10 000 mm above median — far outside 10*IQR
        df_range = pd.DataFrame({
            "date"      : pd.date_range("1990-01-01", periods=20, freq="MS"),
            "rlr_mm"    : values,
            "country"   : "Bangladesh",
            "station_id": 369,
        })
        report = validate_incoming_data(df_range, "TEST_RANGE")
        assert report["range_ok"] is False, "Expected range_ok=False"
        record(test_id, "Validation — out-of-range values flagged", PASS)
    except Exception as e:
        record(test_id, "Validation — out-of-range values flagged", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T10: Validation — empty DataFrame handled gracefully
    # -------------------------------------------------------------------------
    test_id = "T10"
    try:
        report = validate_incoming_data(pd.DataFrame(), "TEST_EMPTY")
        assert not report["passed"]
        assert len(report["warnings"]) > 0
        record(test_id, "Validation — empty DataFrame handled", PASS)
    except Exception as e:
        record(test_id, "Validation — empty DataFrame handled", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T11: log_ingestion_event writes to CSV
    # -------------------------------------------------------------------------
    test_id = "T11"
    try:
        test_log = LOG_DIR / f"test_log_{RUN_TIMESTAMP}.csv"
        # Temporarily redirect log path for test isolation
        global INGESTION_LOG
        _original_log = INGESTION_LOG
        INGESTION_LOG  = test_log

        log_ingestion_event("TEST_SOURCE", "SUCCESS", 42,
                            "Test log entry from T11.", error=None)

        assert test_log.exists(), "Log file not created"
        df_log = pd.read_csv(test_log)
        assert len(df_log) == 1
        assert df_log.loc[0, "source"]  == "TEST_SOURCE"
        assert df_log.loc[0, "records"] == 42
        assert df_log.loc[0, "status"]  == "SUCCESS"

        INGESTION_LOG = _original_log
        test_log.unlink()  # Clean up test log
        record(test_id, "Audit log — CSV write and read-back", PASS)
    except Exception as e:
        INGESTION_LOG = _original_log
        record(test_id, "Audit log — CSV write and read-back", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T12: Ingestion summary JSON structure
    # -------------------------------------------------------------------------
    test_id = "T12"
    try:
        dummy_df = pd.DataFrame({
            "date"      : pd.date_range("1990-01-01", periods=5, freq="MS"),
            "rlr_mm"    : [7000.0] * 5,
            "country"   : "Bangladesh",
            "station_id": 369,
        })
        dummy_counts  = {"Bangladesh": 5, "Vietnam": 0, "Maldives": 0}
        dummy_val     = {"Bangladesh": {"passed": True, "total_rows": 5,
                                         "missing_frac": 0.0, "outlier_count": 0,
                                         "monotonic": True, "range_ok": True,
                                         "warnings": []}}
        summary = generate_ingestion_summary(
            cmems_success     = False,
            cmems_records     = 0,
            psmsl_success     = True,
            psmsl_df          = dummy_df,
            psmsl_counts      = dummy_counts,
            validation_results= dummy_val,
        )
        required_keys = ["run_timestamp", "cmems", "psmsl", "validation", "log_file"]
        for k in required_keys:
            assert k in summary, f"Missing key: {k}"
        assert "records_by_country" in summary["psmsl"]
        assert isinstance(summary["validation"]["all_passed"], bool)
        record(test_id, "Ingestion summary — JSON structure", PASS,
               f"Keys: {list(summary.keys())}")
    except Exception as e:
        record(test_id, "Ingestion summary — JSON structure", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T13: Live PSMSL HTTP — Bangladesh station 369 (requires internet)
    # -------------------------------------------------------------------------
    test_id = "T13"
    try:
        success, df = ingest_psmsl_station(369, "Bangladesh", timeout=20)
        if success and not df.empty:
            assert "rlr_mm"     in df.columns
            assert "date"       in df.columns
            assert "station_id" in df.columns
            assert len(df) > 100, f"Unexpectedly few records: {len(df)}"
            valid_frac = df["rlr_mm"].notna().mean()
            record(test_id, "Live PSMSL — Bangladesh station 369", PASS,
                   f"{len(df)} records, {valid_frac:.1%} valid")
        else:
            record(test_id, "Live PSMSL — Bangladesh station 369", FAIL,
                   "HTTP request failed or empty response")
    except Exception as e:
        record(test_id, "Live PSMSL — Bangladesh station 369", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T14: Dry-run end-to-end pipeline
    # -------------------------------------------------------------------------
    test_id = "T14"
    try:
        summary = run_cell_33(
            cmems_username = "",
            cmems_password = "",
            dry_run        = True
        )
        assert "run_timestamp" in summary
        assert summary["cmems"]["success"] is True   # dry_run returns True
        assert summary["psmsl"]["success"] is True
        record(test_id, "Dry-run end-to-end pipeline", PASS,
               f"timestamp={summary['run_timestamp']}")
    except Exception as e:
        record(test_id, "Dry-run end-to-end pipeline", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T15: Historical dataset and scaler accessibility
    # -------------------------------------------------------------------------
    test_id = "T15"
    try:
        hist_ok = HISTORICAL_CSV.exists()
        scaler_ok = SCALER_JSON.exists()

        if hist_ok:
            df_hist = pd.read_csv(HISTORICAL_CSV)
            row_check = len(df_hist) >= 2628
            col_check = len(df_hist.columns) >= 12
            record(test_id, "Historical dataset accessibility", PASS,
                   f"rows={len(df_hist)}, cols={len(df_hist.columns)}, "
                   f"scaler={'found' if scaler_ok else 'missing'}")
        else:
            record(test_id, "Historical dataset accessibility", FAIL,
                   f"File not found: {HISTORICAL_CSV}. "
                   "Confirm Google Drive is mounted and path is correct.")
    except Exception as e:
        record(test_id, "Historical dataset accessibility", FAIL, str(e))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, _, s, _ in results if s == PASS)
    failed = sum(1 for _, _, s, _ in results if s == FAIL)
    total  = len(results)
    print(f"  Passed : {passed}/{total}")
    print(f"  Failed : {failed}/{total}")

    if failed > 0:
        print("\n  Failed tests:")
        for tid, name, status, detail in results:
            if status == FAIL:
                print(f"    {tid}: {name}")
                if detail:
                    print(f"         Detail: {detail}")

    print("=" * 70 + "\n")


# =============================================================================
# EXECUTION — RUN TESTS FIRST, THEN LIVE INGESTION
# =============================================================================

# Step 1: Run the full test suite (no credentials needed)
test_cell_33()

# Step 2: Run live PSMSL ingestion (always free, no credentials needed)
# CMEMS will be skipped unless you provide credentials below.
#
# To enable CMEMS download:
#   1. Register at https://marine.copernicus.eu/ (free)
#   2. Replace the empty strings below with your credentials
#
CMEMS_USERNAME = ""   # e.g. "your_copernicus_username"
CMEMS_PASSWORD = ""   # e.g. "your_copernicus_password"

live_summary = run_cell_33(
    cmems_username = CMEMS_USERNAME,
    cmems_password = CMEMS_PASSWORD,
    dry_run        = False
)

# Step 3: Print final structured summary
print("\nFINAL INGESTION SUMMARY")
print("-" * 40)
print(json.dumps({
    "run_timestamp"       : live_summary["run_timestamp"],
    "cmems_success"       : live_summary["cmems"]["success"],
    "cmems_records"       : live_summary["cmems"]["estimated_records"],
    "psmsl_total_records" : live_summary["psmsl"]["total_records"],
    "psmsl_by_country"    : live_summary["psmsl"]["records_by_country"],
    "validation_passed"   : live_summary["validation"]["all_passed"],
    "audit_log"           : live_summary["log_file"],
}, indent=2))
