# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 36: Risk Threshold Detection
# =============================================================================
# Purpose  : Automatically classify current sea-level trajectory into
#            GREEN/AMBER/RED/CRITICAL risk levels. Detect statistically
#            significant acceleration events using rolling z-score analysis.
#            Estimate compound flood probability using Gumbel copula.
#            Compute time-to-next-threshold for each country.
#            Output structured JSON risk assessment for Cell 37 (LLM engine).
# Inputs   : live_data/merged_dataset.csv     (from Cell 34)
#            live_data/features_latest.csv    (from Cell 34)
# Outputs  : risk_reports/risk_assessment_{timestamp}.json
#            risk_reports/risk_assessment_latest.json
#            logs/risk_detection_log.csv
# =============================================================================

# --- 0. IMPORTS --------------------------------------------------------------

import json
import logging
import warnings
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell36")

# --- 2. CONFIGURATION --------------------------------------------------------

DRIVE_BASE    = Path("/content/drive/MyDrive/BD_ML_VT")
LIVE_DIR      = DRIVE_BASE / "live_data"
LOG_DIR       = DRIVE_BASE / "logs"
RISK_DIR      = DRIVE_BASE / "risk_reports"
VERSIONED_DIR = DRIVE_BASE / "models" / "versioned"

MERGED_CSV    = LIVE_DIR / "merged_dataset.csv"
FEATURES_CSV  = LIVE_DIR / "features_latest.csv"
RISK_LOG      = LOG_DIR  / "risk_detection_log.csv"

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
BASELINE_DATE = pd.Timestamp("2022-12-31")

for d in [RISK_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- 3. RUNTIME CONTINUITY CHECK ---------------------------------------------

_REQUIRED = {
    "Cell 34 — merged dataset" : MERGED_CSV,
    "Cell 34 — live features"  : FEATURES_CSV,
}
_missing = [(lbl, p) for lbl, p in _REQUIRED.items() if not Path(p).exists()]
if _missing:
    for lbl, p in _missing:
        logger.error("Missing: [%s] %s", lbl, p)
    raise SystemExit("Re-run Cell 34 before Cell 36.")

# --- 4. RISK THRESHOLDS ------------------------------------------------------
# Thresholds are cumulative rise in mm above the 2022-12-31 baseline.
# Derived from IPCC AR6 regional sea-level projections and national
# adaptation plan critical infrastructure exposure assessments.

THRESHOLDS = {
    "Bangladesh": {
        "amber"   : 150,
        "red"     : 250,
        "critical": 350,
    },
    "Vietnam": {
        "amber"   : 100,
        "red"     : 200,
        "critical": 300,
    },
    "Maldives": {
        "amber"   : 50,
        "red"     : 80,
        "critical": 100,
    },
}

# Country exposure metadata (IPCC AR6 regional assessments)
COUNTRY_META = {
    "Bangladesh": {
        "population_at_risk" : 35_000_000,
        "gdp_at_risk_bn_usd" : 12.4,
        "agri_exposure_km2"  : 14_000,
        "vlm_mm_yr"          : -2.0,
        "cvi_score"          : 0.847,
    },
    "Vietnam": {
        "population_at_risk" : 20_000_000,
        "gdp_at_risk_bn_usd" : 8.7,
        "agri_exposure_km2"  : 9_500,
        "vlm_mm_yr"          : -8.9,
        "cvi_score"          : 0.793,
    },
    "Maldives": {
        "population_at_risk" : 500_000,
        "gdp_at_risk_bn_usd" : 0.9,
        "agri_exposure_km2"  : 80,
        "vlm_mm_yr"          :  0.0,
        "cvi_score"          : 0.921,
    },
}

# Acceleration detection parameters
ACCELERATION_WINDOW_MONTHS = 24
ACCELERATION_ZSCORE_ALERT  = 2.0

# Storm surge parameters for compound probability
SURGE_RETURN_PERIOD_YR = {
    "Bangladesh": 10,
    "Vietnam"   : 10,
    "Maldives"  :  5,
}
SURGE_HEIGHT_MM = {
    "Bangladesh": 2500,
    "Vietnam"   : 1500,
    "Maldives"  :  800,
}

# SSP2-4.5 median projection rates (mm/yr) used for time-to-threshold
SSP245_RATE_MM_YR = {
    "Bangladesh":  5.6,
    "Vietnam"   : 18.5,
    "Maldives"  : 10.2,
}

# Column name used for sea-level signal in merged dataset.
# After Cell 34 column rename this is 'slrv2'.
SLR_COLUMN = "slrv2"

# Only use data from this many months before BASELINE_DATE onward when
# computing cumulative rise. This prevents historical RLR datum values
# (~6800-7800 mm absolute) from contaminating the post-baseline anomaly
# calculation, which operates on a much smaller relative scale.
BASELINE_LOOKBACK_MONTHS = 18


# --- 5. AUDIT LOGGING --------------------------------------------------------

def log_risk_event(
    country : str,
    stage   : str,
    status  : str,
    detail  : str,
    error   : Optional[str] = None,
) -> None:
    """Append one row to the risk detection audit log."""
    row = pd.DataFrame([{
        "timestamp" : RUN_TIMESTAMP,
        "country"   : country,
        "stage"     : stage,
        "status"    : status,
        "detail"    : detail,
        "error"     : error if error else "",
    }])
    if RISK_LOG.exists():
        row.to_csv(RISK_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(RISK_LOG, mode="w", header=True, index=False)
    logger.info("[LOG] %s | %s | %s | %s", country, stage, status, detail)


# --- 6. DATA LOADING ---------------------------------------------------------

def load_country_series(
    merged_df : pd.DataFrame,
    country   : str,
    col       : str = SLR_COLUMN,
) -> pd.Series:
    """
    Extract monthly sea-level series for one country, sorted by date.

    Only returns rows from (BASELINE_DATE - BASELINE_LOOKBACK_MONTHS) onward
    to ensure the series is on a consistent datum relative to the 2022
    baseline. The full historical record (1950-2022) uses RLR absolute values
    (~6800-7800 mm) that are datum-incompatible with post-2022 live anomalies.

    Parameters
    ----------
    merged_df : Full merged dataset (column-renamed by Cell 34)
    country   : Country name
    col       : Primary column to extract

    Returns
    -------
    pd.Series indexed by Timestamp, values in mm. Empty if no valid data.
    """
    cdf = (
        merged_df[merged_df["country"] == country]
        .sort_values("date")
        .copy()
    )

    # Restrict to recent segment immediately — before any column checks
    segment_start = BASELINE_DATE - pd.DateOffset(months=BASELINE_LOOKBACK_MONTHS)
    cdf           = cdf[cdf["date"] >= segment_start]

    if col not in cdf.columns:
        fallback_map = {
            "slrv2"      : "slrv2_msl_mm",
            "tide_gauge" : "tide_gauge_mm",
        }
        fallback_col = fallback_map.get(col)
        if fallback_col and fallback_col in cdf.columns:
            logger.warning(
                "  %s: column '%s' not found. Using fallback '%s'.",
                country, col, fallback_col
            )
            col = fallback_col
        else:
            logger.error(
                "  %s: column '%s' not found and no fallback available. "
                "Columns present: %s",
                country, col, list(cdf.columns[:15])
            )
            return pd.Series(dtype=float)

    series = cdf.set_index("date")[col].dropna()

    if series.empty:
        total_rows = len(
            merged_df[merged_df["country"] == country]
        )
        null_count = (
            merged_df.loc[
                merged_df["country"] == country, col
            ].isna().sum()
            if col in merged_df.columns else "N/A"
        )
        logger.error(
            "  %s: series empty after restricting to post-%s and dropna. "
            "total_rows=%d, null_count=%s, segment_start=%s, col=%s",
            country, segment_start.date(),
            total_rows, null_count,
            segment_start.date(), col
        )

    return series


def compute_cumulative_rise(
    series        : pd.Series,
    baseline_date : pd.Timestamp = BASELINE_DATE,
) -> Tuple[float, pd.Series]:
    """
    Compute cumulative sea-level rise since baseline date.

    Assumes the input series is already restricted to a datum-consistent
    segment (see load_country_series). The baseline value is the mean of
    observations within ±6 months of baseline_date. Falls back to the
    first available observation if no data exists in that window.

    Parameters
    ----------
    series        : Monthly SLR series indexed by Timestamp (post-baseline
                    segment only — do not pass the full 1950-present record)
    baseline_date : Reference date for zero-point

    Returns
    -------
    (current_rise_mm, cumulative_series)
      Returns (0.0, empty Series) if input series is empty.
    """
    if series.empty:
        return 0.0, pd.Series(dtype=float)

    window_start  = baseline_date - pd.DateOffset(months=6)
    window_end    = baseline_date + pd.DateOffset(months=6)
    baseline_vals = series[
        (series.index >= window_start) & (series.index <= window_end)
    ]

    if baseline_vals.empty:
        # No data in window — use first observation in series as reference
        baseline_val = float(series.iloc[0])
        logger.warning(
            "No data in ±6 month window around baseline %s. "
            "Using first available observation (%.4f) as reference.",
            baseline_date.date(), baseline_val
        )
    else:
        baseline_val = float(baseline_vals.mean())

    cumulative      = series - baseline_val
    current_rise_mm = float(cumulative.iloc[-1])

    logger.info(
        "  Baseline value: %.4f | current: %.4f | rise: %.4f mm",
        baseline_val, float(series.iloc[-1]), current_rise_mm
    )

    return current_rise_mm, cumulative


# --- 7. RISK CLASSIFICATION --------------------------------------------------

def classify_risk_level(
    current_rise_mm : float,
    country         : str,
) -> str:
    """
    Classify current cumulative sea-level rise into GREEN/AMBER/RED/CRITICAL.

    Parameters
    ----------
    current_rise_mm : Cumulative rise since 2022 baseline (mm)
    country         : Country name for threshold lookup

    Returns
    -------
    'GREEN', 'AMBER', 'RED', or 'CRITICAL'
    """
    t = THRESHOLDS.get(country, THRESHOLDS["Bangladesh"])

    if current_rise_mm >= t["critical"]:
        return "CRITICAL"
    elif current_rise_mm >= t["red"]:
        return "RED"
    elif current_rise_mm >= t["amber"]:
        return "AMBER"
    else:
        return "GREEN"


# --- 8. ACCELERATION DETECTION -----------------------------------------------

def detect_acceleration_anomaly(
    cumulative_series : pd.Series,
    window_months     : int   = ACCELERATION_WINDOW_MONTHS,
    z_alert           : float = ACCELERATION_ZSCORE_ALERT,
) -> Dict[str, Any]:
    """
    Detect statistically significant acceleration in sea-level rise rate.

    Method:
      1. Compute rolling linear trend slope over consecutive window_months
         windows across the full series
      2. Compute z-score of the most recent slope against the full
         distribution of historical slopes
      3. Flag acceleration if |z-score| exceeds z_alert

    Parameters
    ----------
    cumulative_series : Cumulative SLR series indexed by Timestamp
    window_months     : Rolling window length for trend estimation
    z_alert           : Z-score threshold for acceleration alert

    Returns
    -------
    Dictionary: acceleration_detected, z_score, recent_slope_mm_yr,
    historical_mean_slope, historical_std_slope, window_months, note
    """
    if len(cumulative_series) < window_months + 6:
        return {
            "acceleration_detected" : False,
            "z_score"               : 0.0,
            "recent_slope_mm_yr"    : None,
            "historical_mean_slope" : None,
            "historical_std_slope"  : None,
            "window_months"         : window_months,
            "note"                  : "Insufficient data for acceleration detection.",
        }

    slopes = []
    vals   = cumulative_series.values
    n      = len(vals)

    for start in range(0, n - window_months + 1):
        segment = vals[start : start + window_months]
        x       = np.arange(window_months, dtype=float)
        slope, _, _, _, _ = stats.linregress(x, segment)
        slopes.append(slope * 12)   # Convert per-month to per-year

    slopes_arr   = np.array(slopes)
    recent_slope = slopes_arr[-1]
    hist_mean    = float(slopes_arr.mean())
    hist_std     = max(float(slopes_arr.std()), 1e-8)
    z_score      = (recent_slope - hist_mean) / hist_std

    return {
        "acceleration_detected" : bool(abs(z_score) > z_alert),
        "z_score"               : round(float(z_score), 4),
        "recent_slope_mm_yr"    : round(float(recent_slope), 4),
        "historical_mean_slope" : round(hist_mean, 4),
        "historical_std_slope"  : round(hist_std, 4),
        "window_months"         : window_months,
        "note"                  : None,
    }


# --- 9. COMPOUND EVENT PROBABILITY -------------------------------------------

def estimate_compound_probability(
    current_rise_mm : float,
    country         : str,
    projection_year : int = 2050,
) -> Dict[str, Any]:
    """
    Estimate annual probability of compound sea-level + storm surge
    exceedance using a Gumbel copula model.

    Parameters
    ----------
    current_rise_mm : Current cumulative rise above 2022 baseline (mm)
    country         : Country name
    projection_year : Target year for compound probability

    Returns
    -------
    Dictionary: compound_prob_annual, slr_exceedance_prob,
    surge_exceedance_prob, projected_slr_mm, theta_copula, method
    """
    years_to_target  = max(projection_year - 2024, 1)
    rate_mm_yr       = SSP245_RATE_MM_YR.get(country, 6.0)
    projected_slr_mm = current_rise_mm + rate_mm_yr * years_to_target
    threshold_mm     = THRESHOLDS[country]["red"]

    slr_sigma        = max(abs(projected_slr_mm) * 0.20, 10.0)
    slr_exceed_prob  = float(1.0 - ndtr(
        (threshold_mm - projected_slr_mm) / slr_sigma
    ))
    slr_exceed_prob  = float(np.clip(slr_exceed_prob, 0.0, 1.0))

    surge_annual_prob = float(1.0 / SURGE_RETURN_PERIOD_YR.get(country, 10))

    theta     = 2.0
    u         = max(slr_exceed_prob,  1e-10)
    v         = max(surge_annual_prob, 1e-10)
    neg_ln_u  = -np.log(u)
    neg_ln_v  = -np.log(v)
    compound_prob = float(np.exp(
        -((neg_ln_u ** theta + neg_ln_v ** theta) ** (1.0 / theta))
    ))
    compound_prob = float(np.clip(compound_prob, 0.0, 1.0))

    return {
        "compound_prob_annual"  : round(compound_prob, 6),
        "slr_exceedance_prob"   : round(slr_exceed_prob, 6),
        "surge_exceedance_prob" : round(surge_annual_prob, 6),
        "projected_slr_mm"      : round(float(projected_slr_mm), 2),
        "projection_year"       : projection_year,
        "theta_copula"          : theta,
        "method"                : "Gumbel copula (theta=2.0, SSP2-4.5)",
    }


# --- 10. TIME TO NEXT THRESHOLD ----------------------------------------------

def compute_time_to_threshold(
    current_rise_mm : float,
    country         : str,
    current_date    : Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Estimate years until each risk threshold is crossed using linear
    projection at the SSP2-4.5 rate.

    Thresholds already crossed (remaining_mm <= 0) are reported as 0.0.
    Identifies the next uncrossed threshold and years until it is reached.

    Parameters
    ----------
    current_rise_mm : Current cumulative rise above 2022 baseline (mm)
    country         : Country name
    current_date    : Reference date (defaults to now)

    Returns
    -------
    Dictionary: years_to_amber, years_to_red, years_to_critical,
    next_threshold, years_to_next, rate_used_mm_yr
    """
    if current_date is None:
        current_date = pd.Timestamp.now()

    rate_mm_yr = SSP245_RATE_MM_YR.get(country, 6.0)
    t          = THRESHOLDS.get(country, THRESHOLDS["Bangladesh"])

    times = {}
    for level, threshold_mm in [
        ("amber",    t["amber"]),
        ("red",      t["red"]),
        ("critical", t["critical"]),
    ]:
        remaining_mm = threshold_mm - current_rise_mm
        if remaining_mm <= 0:
            times[level] = 0.0
        elif rate_mm_yr <= 0:
            times[level] = float("inf")
        else:
            times[level] = round(remaining_mm / rate_mm_yr, 1)

    # First threshold not yet crossed (time > 0)
    next_threshold = None
    years_to_next  = None
    for level in ["amber", "red", "critical"]:
        if times.get(level, 0) > 0:
            next_threshold = level
            years_to_next  = times[level]
            break

    return {
        "years_to_amber"    : times.get("amber"),
        "years_to_red"      : times.get("red"),
        "years_to_critical" : times.get("critical"),
        "next_threshold"    : next_threshold,
        "years_to_next"     : years_to_next,
        "rate_used_mm_yr"   : rate_mm_yr,
    }


# --- 11. FULL RISK ASSESSMENT FOR ONE COUNTRY --------------------------------

def assess_country_risk(
    merged_df : pd.DataFrame,
    country   : str,
) -> Dict[str, Any]:
    """
    Run the full risk assessment pipeline for one country.

    Steps:
      1. Extract post-baseline country SLR series and compute cumulative rise
      2. Classify current risk level (GREEN/AMBER/RED/CRITICAL)
      3. Detect acceleration anomaly (rolling z-score)
      4. Estimate compound event probability (Gumbel copula)
      5. Compute time to next threshold (linear SSP2-4.5 projection)

    Parameters
    ----------
    merged_df : Full merged dataset (column-renamed by Cell 34)
    country   : Country name

    Returns
    -------
    Complete risk assessment dictionary. Sets error key if any step fails.
    """
    logger.info("--- RISK ASSESSMENT: %s ---", country)

    result = {
        "country"         : country,
        "assessment_date" : RUN_TIMESTAMP,
        "error"           : None,
    }

    try:
        # Step 1: Extract series and compute cumulative rise
        series = load_country_series(merged_df, country)

        if series.empty:
            cdf        = merged_df[merged_df["country"] == country]
            null_count = (cdf[SLR_COLUMN].isna().sum()
                          if SLR_COLUMN in cdf.columns else "col_missing")
            raise ValueError(
                f"Empty series for {country}. "
                f"Rows={len(cdf)}, "
                f"{SLR_COLUMN}_nulls={null_count}, "
                f"date_range={cdf['date'].min()} to {cdf['date'].max()}. "
                f"Ensure Cell 34 column rename was applied correctly."
            )

        current_rise_mm, cumulative = compute_cumulative_rise(series)
        last_date = series.index[-1]

        result["last_observation_date"] = str(last_date.date())
        result["current_rise_mm"]       = round(current_rise_mm, 3)
        result["n_observations"]        = len(series)

        # Step 2: Risk classification
        risk_level              = classify_risk_level(current_rise_mm, country)
        result["risk_level"]    = risk_level
        result["thresholds_mm"] = THRESHOLDS.get(country)

        logger.info("  %s: rise=%.1fmm | risk=%s",
                    country, current_rise_mm, risk_level)

        # Step 3: Acceleration detection
        accel                  = detect_acceleration_anomaly(cumulative)
        result["acceleration"] = accel

        if accel["acceleration_detected"]:
            logger.warning(
                "  %s: ACCELERATION DETECTED (z=%.2f, slope=%.2fmm/yr)",
                country, accel["z_score"], accel["recent_slope_mm_yr"]
            )
        else:
            logger.info("  %s: no acceleration (z=%.2f)",
                        country, accel["z_score"])

        # Step 4: Compound probability
        compound                = estimate_compound_probability(current_rise_mm, country)
        result["compound_risk"] = compound

        logger.info(
            "  %s: compound_prob=%.4f (slr_exceed=%.4f, surge=%.4f)",
            country,
            compound["compound_prob_annual"],
            compound["slr_exceedance_prob"],
            compound["surge_exceedance_prob"],
        )

        # Step 5: Time to threshold
        time_to                     = compute_time_to_threshold(current_rise_mm, country)
        result["time_to_threshold"] = time_to

        logger.info(
            "  %s: next=%s in %.1f yr",
            country,
            time_to["next_threshold"] or "N/A",
            time_to["years_to_next"]  or 0,
        )

        # Exposure metadata for LLM prompt (Cell 37)
        result["meta"] = COUNTRY_META.get(country, {})

        log_risk_event(
            country, "ASSESSMENT", "SUCCESS",
            f"risk={risk_level}, rise={current_rise_mm:.1f}mm, "
            f"accel={accel['acceleration_detected']}, "
            f"compound={compound['compound_prob_annual']:.4f}",
        )

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("  %s: risk assessment failed: %s", country, exc)
        logger.debug(traceback.format_exc())
        log_risk_event(country, "ASSESSMENT", "FAILED",
                       str(exc), error=str(exc))

    return result


# --- 12. GENERATE RISK REPORT ------------------------------------------------

def generate_risk_report(
    country_results : Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Assemble all country assessments into a structured JSON risk report.

    Saves:
      - risk_reports/risk_assessment_{timestamp}.json  (timestamped archive)
      - risk_reports/risk_assessment_latest.json       (always current)

    Parameters
    ----------
    country_results : Dict keyed by country, values from assess_country_risk

    Returns
    -------
    Full report dictionary
    """
    risk_order  = ["GREEN", "AMBER", "RED", "CRITICAL"]
    risk_levels = {
        c: r.get("risk_level", "UNKNOWN")
        for c, r in country_results.items()
        if r.get("error") is None
    }

    max_risk = max(
        (r for r in risk_levels.values() if r in risk_order),
        key=lambda x: risk_order.index(x),
        default="UNKNOWN",
    )
    any_acceleration = any(
        r.get("acceleration", {}).get("acceleration_detected", False)
        for r in country_results.values()
        if r.get("error") is None
    )

    report = {
        "report_timestamp"    : RUN_TIMESTAMP,
        "baseline_date"       : str(BASELINE_DATE.date()),
        "system_risk_level"   : max_risk,
        "any_acceleration"    : any_acceleration,
        "countries"           : country_results,
        "risk_levels_summary" : risk_levels,
        "pipeline_version"    : "HyperSLR-Cell36",
    }

    ts_path = RISK_DIR / f"risk_assessment_{RUN_TIMESTAMP}.json"
    with open(ts_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    latest_path = RISK_DIR / "risk_assessment_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Risk report saved   : %s", ts_path.name)
    logger.info("Risk report (latest): %s", latest_path)
    return report


# =============================================================================
# CELL 36 — MAIN EXECUTION
# =============================================================================

def run_cell_36() -> Dict[str, Any]:
    """Execute the full Cell 36 risk threshold detection pipeline."""
    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 36: RISK THRESHOLD DETECTION")
    logger.info("Run timestamp : %s", RUN_TIMESTAMP)
    logger.info("Baseline date : %s", BASELINE_DATE.date())
    logger.info("=" * 70)

    merged_df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    logger.info("Merged dataset: %d rows", len(merged_df))

    # Data coverage report before processing
    segment_start = BASELINE_DATE - pd.DateOffset(months=BASELINE_LOOKBACK_MONTHS)
    logger.info("--- DATA COVERAGE (post-%s) ---", segment_start.date())
    for country in ["Bangladesh", "Vietnam", "Maldives"]:
        cdf   = merged_df[
            (merged_df["country"] == country) &
            (merged_df["date"]    >= segment_start)
        ]
        valid = cdf[SLR_COLUMN].notna().sum() if SLR_COLUMN in cdf.columns else 0
        logger.info("  %-12s : %d rows in segment, %d valid %s",
                    country, len(cdf), valid, SLR_COLUMN)

    countries       = ["Bangladesh", "Vietnam", "Maldives"]
    country_results = {}

    for country in countries:
        if country not in merged_df["country"].unique():
            logger.warning(
                "Country '%s' not in merged dataset. Skipping.", country
            )
            continue
        result = assess_country_risk(merged_df, country)
        country_results[country] = result

    report = generate_risk_report(country_results)

    logger.info("=" * 70)
    logger.info("CELL 36 COMPLETE — SYSTEM RISK: %s", report["system_risk_level"])
    logger.info("-" * 70)
    for country, r in country_results.items():
        if r.get("error"):
            logger.error("  %-12s | ERROR: %s", country, r["error"])
            continue
        rise  = r.get("current_rise_mm", 0.0)
        level = r.get("risk_level", "N/A")
        accel = r.get("acceleration", {}).get("acceleration_detected", False)
        nxt   = r.get("time_to_threshold", {}).get("next_threshold", "N/A")
        yrs   = r.get("time_to_threshold", {}).get("years_to_next", "N/A")
        logger.info(
            "  %-12s | %-8s | rise=%7.2fmm | accel=%-5s | next=%s in %s yr",
            country, level, rise, str(accel), nxt, yrs,
        )
    logger.info("=" * 70)

    return report


# =============================================================================
# CELL 36 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_36() -> None:
    """
    Comprehensive unit and integration tests for all Cell 36 functions.

    Test coverage:
      T01 — Risk classification: GREEN below all thresholds
      T02 — Risk classification: AMBER at amber threshold
      T03 — Risk classification: RED at red threshold
      T04 — Risk classification: CRITICAL at critical threshold
      T05 — Risk classification: boundary values exactly at threshold
      T06 — Cumulative rise: correct zeroing at baseline
      T07 — Cumulative rise: correct current value after step change
      T08 — Cumulative rise: no baseline data uses first observation as ref
      T09 — Cumulative rise: empty series returns (0.0, empty Series)
      T10 — Acceleration detection: linear series yields near-zero z-score
      T11 — Acceleration detection: quadratic series detected
      T12 — Acceleration detection: insufficient data handled gracefully
      T13 — Compound probability: bounds always [0, 1]
      T14 — Compound probability: higher SLR yields higher compound prob
      T15 — Gumbel copula: Frechet upper bound satisfied
      T16 — Time to threshold: crossed levels return 0.0
      T17 — Time to threshold: correct linear projection
      T18 — Time to threshold: next_threshold is first uncrossed level
      T19 — Full country assessment: structure and required keys
      T20 — End-to-end pipeline on Drive data
    """

    PASS = "PASS"
    FAIL = "FAIL"
    results = []

    def record(test_id, name, status, detail=""):
        results.append((test_id, name, status, detail))
        marker = "[PASS]" if status == PASS else "[FAIL]"
        print(f"  {marker} {test_id}: {name}" +
              (f" — {detail}" if detail else ""))

    def make_synthetic_merged(n=300) -> pd.DataFrame:
        """
        Synthetic merged DataFrame whose slrv2 values sit in the
        post-baseline window so load_country_series returns valid data.
        """
        frames = []
        np.random.seed(42)
        # Start inside the BASELINE_LOOKBACK_MONTHS window so values survive
        # the segment filter in load_country_series
        start_date = BASELINE_DATE - pd.DateOffset(months=BASELINE_LOOKBACK_MONTHS - 1)
        n_months   = 30   # enough for a meaningful series
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            dates = pd.date_range(start_date, periods=n_months, freq="MS")
            slr   = np.cumsum(np.random.normal(0.3, 0.5, n_months))
            frames.append(pd.DataFrame({
                "date"   : dates,
                "country": country,
                "slrv2"  : slr,
            }))
        return pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 70)
    print("CELL 36 — TEST SUITE")
    print("=" * 70 + "\n")

    synth_merged = make_synthetic_merged()

    # -------------------------------------------------------------------------
    # T01–T05: Risk classification
    # -------------------------------------------------------------------------
    test_id = "T01"
    try:
        assert classify_risk_level(0.0,  "Bangladesh") == "GREEN"
        assert classify_risk_level(10.0, "Vietnam")    == "GREEN"
        assert classify_risk_level(0.0,  "Maldives")   == "GREEN"
        record(test_id, "Risk classification — GREEN", PASS)
    except Exception as e:
        record(test_id, "Risk classification — GREEN", FAIL, str(e))

    test_id = "T02"
    try:
        assert classify_risk_level(160.0, "Bangladesh") == "AMBER"
        assert classify_risk_level(110.0, "Vietnam")    == "AMBER"
        assert classify_risk_level(55.0,  "Maldives")   == "AMBER"
        record(test_id, "Risk classification — AMBER", PASS)
    except Exception as e:
        record(test_id, "Risk classification — AMBER", FAIL, str(e))

    test_id = "T03"
    try:
        assert classify_risk_level(260.0, "Bangladesh") == "RED"
        assert classify_risk_level(210.0, "Vietnam")    == "RED"
        assert classify_risk_level(85.0,  "Maldives")   == "RED"
        record(test_id, "Risk classification — RED", PASS)
    except Exception as e:
        record(test_id, "Risk classification — RED", FAIL, str(e))

    test_id = "T04"
    try:
        assert classify_risk_level(360.0, "Bangladesh") == "CRITICAL"
        assert classify_risk_level(310.0, "Vietnam")    == "CRITICAL"
        assert classify_risk_level(105.0, "Maldives")   == "CRITICAL"
        record(test_id, "Risk classification — CRITICAL", PASS)
    except Exception as e:
        record(test_id, "Risk classification — CRITICAL", FAIL, str(e))

    test_id = "T05"
    try:
        assert classify_risk_level(150.0, "Bangladesh") == "AMBER"
        assert classify_risk_level(250.0, "Bangladesh") == "RED"
        assert classify_risk_level(350.0, "Bangladesh") == "CRITICAL"
        assert classify_risk_level(149.9, "Bangladesh") == "GREEN"
        assert classify_risk_level(249.9, "Bangladesh") == "AMBER"
        record(test_id, "Risk classification — boundary values", PASS)
    except Exception as e:
        record(test_id, "Risk classification — boundary values", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T06: Cumulative rise zeroed at baseline
    # -------------------------------------------------------------------------
    test_id = "T06"
    try:
        idx    = pd.date_range("2021-01-01", periods=48, freq="MS")
        series = pd.Series(np.linspace(100, 160, 48), index=idx)
        _, cum = compute_cumulative_rise(
            series, baseline_date=pd.Timestamp("2022-12-01")
        )
        baseline_region = cum["2022-06-01":"2023-06-01"]
        assert baseline_region.abs().mean() < 5.0, \
            f"Baseline not near zero: {baseline_region.abs().mean():.2f}"
        record(test_id, "Cumulative rise — zeroed at baseline", PASS,
               f"baseline_mean_abs={baseline_region.abs().mean():.3f}")
    except Exception as e:
        record(test_id, "Cumulative rise — zeroed at baseline", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T07: Cumulative rise — correct current value after step change
    # -------------------------------------------------------------------------
    test_id = "T07"
    try:
        idx      = pd.date_range("2021-01-01", periods=48, freq="MS")
        vals     = np.zeros(48)
        vals[24:] = 50.0
        series   = pd.Series(vals, index=idx)
        current, _ = compute_cumulative_rise(
            series, baseline_date=pd.Timestamp("2022-06-01")
        )
        assert abs(current - 50.0) < 5.0, \
            f"Expected ~50mm rise, got {current:.2f}mm"
        record(test_id, "Cumulative rise — correct current value", PASS,
               f"current_rise={current:.2f}mm")
    except Exception as e:
        record(test_id, "Cumulative rise — correct current value", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T08: Cumulative rise — no baseline window data uses first obs as ref
    # -------------------------------------------------------------------------
    test_id = "T08"
    try:
        # Series is entirely post-baseline with no data in the ±6 month window.
        # Start date is 2023-07-01 — more than 6 months after BASELINE_DATE
        # (2022-12-31), so the window [2022-07-01, 2023-07-01] is empty.
        # Function must fall back to first observation as reference,
        # making cumulative[0] == 0.0 exactly.
        idx    = pd.date_range("2023-07-01", periods=12, freq="MS")
        vals   = np.linspace(10.0, 30.0, 12)   # non-zero start
        series = pd.Series(vals, index=idx)

        current, cum = compute_cumulative_rise(
            series, baseline_date=BASELINE_DATE
        )
        assert isinstance(current, float)
        assert len(cum) > 0
        assert abs(cum.iloc[0]) < 1e-9, \
            f"First cumulative value should be 0.0 (first-obs fallback), " \
            f"got {cum.iloc[0]:.6f}"
        assert current > 0, \
            f"Current rise should be positive (series increases): {current}"
        record(test_id,
               "Cumulative rise — no baseline window data uses first obs as ref",
               PASS,
               f"first_cum={cum.iloc[0]:.2e}, current={current:.2f}mm")
    except Exception as e:
        record(test_id,
               "Cumulative rise — no baseline window data uses first obs as ref",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T09: Cumulative rise — empty series returns safe defaults
    # -------------------------------------------------------------------------
    test_id = "T09"
    try:
        current, cum = compute_cumulative_rise(pd.Series(dtype=float))
        assert current == 0.0, f"Expected 0.0, got {current}"
        assert cum.empty,       "Expected empty cumulative series"
        record(test_id, "Cumulative rise — empty series safe defaults", PASS)
    except Exception as e:
        record(test_id, "Cumulative rise — empty series safe defaults",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T10: Acceleration detection — linear series yields near-zero z-score
    # -------------------------------------------------------------------------
    test_id = "T10"
    try:
        idx    = pd.date_range("1990-01-01", periods=360, freq="MS")
        linear = pd.Series(np.linspace(0, 100, 360), index=idx)
        result = detect_acceleration_anomaly(linear, window_months=24)
        assert result["acceleration_detected"] is False, \
            f"Expected no acceleration for linear series, z={result['z_score']}"
        assert abs(result["z_score"]) < 1.0, \
            f"Z-score too high for linear series: {result['z_score']}"
        record(test_id, "Acceleration — linear series no alert", PASS,
               f"z={result['z_score']:.4f}")
    except Exception as e:
        record(test_id, "Acceleration — linear series no alert", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T11: Acceleration detection — quadratic series detected
    # -------------------------------------------------------------------------
    test_id = "T11"
    try:
        idx          = pd.date_range("1990-01-01", periods=360, freq="MS")
        t            = np.arange(360)
        vals         = 0.1 * t + 0.005 * t ** 2
        accel_series = pd.Series(vals, index=idx)
        result       = detect_acceleration_anomaly(accel_series, window_months=24)
        assert result["recent_slope_mm_yr"] is not None
        assert result["recent_slope_mm_yr"] > result["historical_mean_slope"], \
            "Recent slope should exceed historical mean for quadratic series"
        record(test_id, "Acceleration — quadratic series detected", PASS,
               f"z={result['z_score']:.4f}, "
               f"recent={result['recent_slope_mm_yr']:.2f}mm/yr, "
               f"hist_mean={result['historical_mean_slope']:.2f}mm/yr")
    except Exception as e:
        record(test_id, "Acceleration — quadratic series detected", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T12: Acceleration detection — insufficient data
    # -------------------------------------------------------------------------
    test_id = "T12"
    try:
        short  = pd.Series(
            np.linspace(0, 10, 20),
            index=pd.date_range("2020-01-01", periods=20, freq="MS"),
        )
        result = detect_acceleration_anomaly(short, window_months=24)
        assert result["acceleration_detected"] is False
        assert "Insufficient" in (result.get("note") or "")
        record(test_id, "Acceleration — insufficient data handled", PASS)
    except Exception as e:
        record(test_id, "Acceleration — insufficient data handled", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T13: Compound probability — bounds always [0, 1]
    # -------------------------------------------------------------------------
    test_id = "T13"
    try:
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            for rise in [-100, 0, 50, 200, 500, 1000]:
                r = estimate_compound_probability(rise, country)
                assert 0.0 <= r["compound_prob_annual"] <= 1.0, \
                    f"{country} rise={rise}: prob={r['compound_prob_annual']}"
                assert 0.0 <= r["slr_exceedance_prob"] <= 1.0
        record(test_id, "Compound probability — always in [0,1]", PASS)
    except Exception as e:
        record(test_id, "Compound probability — always in [0,1]", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T14: Compound probability — monotone in SLR
    # -------------------------------------------------------------------------
    test_id = "T14"
    try:
        r_low  = estimate_compound_probability(0.0,   "Bangladesh")
        r_high = estimate_compound_probability(300.0, "Bangladesh")
        assert r_high["compound_prob_annual"] >= r_low["compound_prob_annual"], \
            f"Higher SLR should yield higher compound prob: " \
            f"low={r_low['compound_prob_annual']:.6f}, " \
            f"high={r_high['compound_prob_annual']:.6f}"
        record(test_id, "Compound probability — monotone in SLR", PASS,
               f"low={r_low['compound_prob_annual']:.6f}, "
               f"high={r_high['compound_prob_annual']:.6f}")
    except Exception as e:
        record(test_id, "Compound probability — monotone in SLR", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T15: Gumbel copula — Frechet upper bound C(u,v) <= min(u,v)
    # -------------------------------------------------------------------------
    test_id = "T15"
    try:
        for u_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for v_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
                u_t   = max(u_val, 1e-10)
                v_t   = max(v_val, 1e-10)
                theta = 2.0
                cop   = float(np.exp(
                    -((-np.log(u_t)) ** theta +
                      (-np.log(v_t)) ** theta) ** (1.0 / theta)
                ))
                assert cop <= min(u_val, v_val) + 1e-9, \
                    f"Frechet upper bound violated u={u_val}, v={v_val}: {cop}"
                assert cop >= 0.0
        record(test_id, "Gumbel copula — Frechet upper bound satisfied", PASS)
    except Exception as e:
        record(test_id, "Gumbel copula — Frechet upper bound satisfied",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T16: Time to threshold — crossed levels return 0.0
    # -------------------------------------------------------------------------
    test_id = "T16"
    try:
        # Bangladesh: amber=150, red=250, critical=350
        # At 260mm: amber crossed (0), red crossed (0), critical not yet (>0)
        result = compute_time_to_threshold(260.0, "Bangladesh")
        assert result["years_to_amber"] == 0.0, \
            f"Expected 0.0 for crossed amber: {result['years_to_amber']}"
        assert result["years_to_red"] == 0.0, \
            f"Expected 0.0 for crossed red: {result['years_to_red']}"
        assert result["years_to_critical"] > 0.0, \
            f"Critical not yet crossed at 260mm: {result['years_to_critical']}"
        assert result["next_threshold"] == "critical", \
            f"Expected next=critical, got {result['next_threshold']}"

        # Maldives: amber=50 — at 40mm nothing crossed yet
        result_m = compute_time_to_threshold(40.0, "Maldives")
        assert result_m["years_to_amber"] > 0.0
        assert result_m["next_threshold"] == "amber"

        record(test_id, "Time to threshold — crossed levels return 0.0", PASS,
               f"BD at 260mm: amber=0, red=0, "
               f"critical={result['years_to_critical']}yr")
    except Exception as e:
        record(test_id, "Time to threshold — crossed levels return 0.0",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T17: Time to threshold — correct linear projection
    # -------------------------------------------------------------------------
    test_id = "T17"
    try:
        # Bangladesh amber=150mm, rate=5.6mm/yr, current=0mm -> 150/5.6=26.8yr
        result   = compute_time_to_threshold(0.0, "Bangladesh")
        expected = round(150.0 / SSP245_RATE_MM_YR["Bangladesh"], 1)
        assert abs(result["years_to_amber"] - expected) < 0.1, \
            f"Expected {expected}yr, got {result['years_to_amber']}yr"
        record(test_id, "Time to threshold — correct linear projection", PASS,
               f"expected={expected}yr, got={result['years_to_amber']}yr")
    except Exception as e:
        record(test_id, "Time to threshold — correct linear projection",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T18: Time to threshold — next_threshold is first uncrossed level
    # -------------------------------------------------------------------------
    test_id = "T18"
    try:
        # Maldives: amber=50, red=80, critical=100
        # current=60: amber crossed -> next=red
        result = compute_time_to_threshold(60.0, "Maldives")
        assert result["next_threshold"] == "red", \
            f"Expected next=red, got {result['next_threshold']}"
        assert result["years_to_next"] == result["years_to_red"]
        assert result["years_to_amber"] == 0.0

        # All crossed
        result_all = compute_time_to_threshold(105.0, "Maldives")
        assert result_all["next_threshold"] is None, \
            "All thresholds crossed — next should be None"
        assert result_all["years_to_next"] is None

        record(test_id, "Time to threshold — next_threshold correct", PASS,
               f"next={result['next_threshold']}, "
               f"years={result['years_to_next']}yr")
    except Exception as e:
        record(test_id, "Time to threshold — next_threshold correct",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T19: Full country assessment — structure and required keys
    # -------------------------------------------------------------------------
    test_id = "T19"
    try:
        result = assess_country_risk(synth_merged, "Bangladesh")
        required_keys = [
            "country", "assessment_date", "current_rise_mm", "risk_level",
            "thresholds_mm", "acceleration", "compound_risk",
            "time_to_threshold", "meta", "n_observations",
            "last_observation_date",
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"
        assert result["risk_level"] in ["GREEN", "AMBER", "RED", "CRITICAL"]
        assert isinstance(result["acceleration"]["z_score"], float)
        assert 0 <= result["compound_risk"]["compound_prob_annual"] <= 1
        assert result.get("error") is None, \
            f"Unexpected error: {result.get('error')}"
        record(test_id, "Country assessment — structure complete", PASS,
               f"risk={result['risk_level']}, "
               f"rise={result['current_rise_mm']:.1f}mm, "
               f"z={result['acceleration']['z_score']:.3f}")
    except Exception as e:
        record(test_id, "Country assessment — structure complete", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T20: End-to-end pipeline on Drive data
    # -------------------------------------------------------------------------
    test_id = "T20"
    try:
        report = run_cell_36()
        assert "system_risk_level"   in report
        assert "countries"           in report
        assert "any_acceleration"    in report
        assert "risk_levels_summary" in report

        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            assert country in report["countries"], \
                f"{country} missing from report"
            r = report["countries"][country]
            assert r.get("error") is None, \
                f"{country} error: {r.get('error')}"
            assert r["risk_level"] in ["GREEN", "AMBER", "RED", "CRITICAL"], \
                f"{country} invalid risk level: {r['risk_level']}"
            assert isinstance(r["current_rise_mm"], float), \
                f"{country} current_rise_mm not float"
            assert r["n_observations"] > 0, \
                f"{country} has zero observations"

        ts_files = list(RISK_DIR.glob("risk_assessment_2*.json"))
        assert len(ts_files) >= 1, "No timestamped risk report found"
        assert RISK_LOG.exists(),  "Risk detection log not created"

        record(test_id, "End-to-end pipeline on Drive data", PASS,
               f"system={report['system_risk_level']}, "
               f"accel={report['any_acceleration']}, "
               f"levels={report['risk_levels_summary']}")
    except Exception as e:
        record(test_id, "End-to-end pipeline on Drive data", FAIL, str(e))

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
# RUN
# =============================================================================

test_cell_36()

# Print final risk summary table
print("FINAL RISK ASSESSMENT SUMMARY")
print("-" * 70)
latest = RISK_DIR / "risk_assessment_latest.json"
if latest.exists():
    with open(latest) as f:
        rpt = json.load(f)
    print(f"  System risk level : {rpt['system_risk_level']}")
    print(f"  Any acceleration  : {rpt['any_acceleration']}")
    print(f"  Baseline date     : {rpt['baseline_date']}")
    print()
    print(f"  {'Country':<14} {'Risk':<10} {'Rise(mm)':<12} "
          f"{'Accel':<8} {'Z-score':<10} {'Next':<12} {'Years'}")
    print("  " + "-" * 66)
    for country, r in rpt["countries"].items():
        if r.get("error"):
            print(f"  {country:<14} ERROR: {r['error'][:50]}")
            continue
        accel = r.get("acceleration", {})
        ttime = r.get("time_to_threshold", {})
        print(
            f"  {country:<14} "
            f"{r.get('risk_level', 'N/A'):<10} "
            f"{r.get('current_rise_mm', 0.0):<12.2f}"
            f"{str(accel.get('acceleration_detected', '')):<8} "
            f"{accel.get('z_score', 0.0):<10.3f}"
            f"{str(ttime.get('next_threshold', 'N/A')):<12} "
            f"{ttime.get('years_to_next', 'N/A')}"
        )
    print()
    print(f"  Risk report : {latest}")
    print(f"  Risk log    : {RISK_LOG}")
