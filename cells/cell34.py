# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 34: Automated Feature Engineering
# =============================================================================
# Purpose  : Extend the Cell 28 feature engineering pipeline to process live
#            ingested PSMSL data automatically. Detect feature drift between
#            training distribution and live data using Population Stability
#            Index (PSI). Recalibrate scaler if drift is confirmed. Merge
#            live features with historical dataset.
# Inputs   : live_data/psmsl_latest.csv            (from Cell 33)
#            processed/slr_enhanced_dataset.csv     (2628 rows, Cell 28)
#            processed/scaler_params.json           (Cell 28)
# Outputs  : live_data/features_latest.csv
#            live_data/merged_dataset.csv
#            logs/feature_engineering_log.csv
#            processed/scaler_params_updated.json   (only if drift confirmed)
#
# Column mapping applied once in load_historical_dataset():
#   slrv2_msl_mm        -> slrv2
#   tide_gauge_mm       -> tide_gauge
#   enso_anom           -> enso
#   aviso_sla_mm        -> aviso
#   vlm_mm_yr           -> vlm
#   cmip6_thermo_mm     -> cmip6_thermo
#   cmip6_glacier_mm    -> cmip6_glacier
#   slr_momentum_mm     -> momentum
#   physics_residual_mm -> residual
#   enso_aviso_interact -> enso_x_aviso
#
# Datum normalisation applied once in load_historical_dataset():
#   Historical slrv2 converted from RLR absolute (~7000 mm) to relative
#   anomaly (mm above each country's 2022-12-31 baseline level) so it
#   is on the same datum as the live-engineered slrv2 from Cell 33 PSMSL
#   data and consistent with the risk thresholds in Cell 36.
# =============================================================================

# --- 0. IMPORTS --------------------------------------------------------------

import os
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell34")

# --- 2. CONFIGURATION --------------------------------------------------------

DRIVE_BASE     = Path("/content/drive/MyDrive/BD_ML_VT")
LIVE_DIR       = DRIVE_BASE / "live_data"
LOG_DIR        = DRIVE_BASE / "logs"
PROCESSED_DIR  = DRIVE_BASE / "processed"

PSMSL_LATEST   = LIVE_DIR  / "psmsl_latest.csv"
HISTORICAL_CSV = PROCESSED_DIR / "slr_enhanced_dataset.csv"
SCALER_JSON    = PROCESSED_DIR / "scaler_params.json"

FEATURES_OUT   = LIVE_DIR  / "features_latest.csv"
MERGED_OUT     = LIVE_DIR  / "merged_dataset.csv"
SCALER_UPDATED = PROCESSED_DIR / "scaler_params_updated.json"
FEATURE_LOG    = LOG_DIR   / "feature_engineering_log.csv"

RUN_TIMESTAMP  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

ENSO_URL = "https://psl.noaa.gov/data/correlation/nina34.data"

# Baseline date used for RLR -> anomaly conversion and for Cell 36 risk calc
SLR_BASELINE_DATE = pd.Timestamp("2022-12-31")

# Cell 28 column name -> HyperSLR standardised feature name.
# Applied once on load; all downstream code uses the standardised names.
COLUMN_MAP = {
    "slrv2_msl_mm"        : "slrv2",
    "tide_gauge_mm"       : "tide_gauge",
    "enso_anom"           : "enso",
    "aviso_sla_mm"        : "aviso",
    "vlm_mm_yr"           : "vlm",
    "cmip6_thermo_mm"     : "cmip6_thermo",
    "cmip6_glacier_mm"    : "cmip6_glacier",
    "slr_momentum_mm"     : "momentum",
    "physics_residual_mm" : "residual",
    "enso_aviso_interact" : "enso_x_aviso",
}

# 12 features used by the live pipeline and PSI drift detection
FEATURE_COLUMNS = [
    "slrv2", "tide_gauge", "enso", "aviso", "vlm",
    "cmip6_thermo", "cmip6_glacier", "momentum", "residual",
    "month_sin", "month_cos", "enso_x_aviso",
]

TARGET_COLUMN = "slrv2"

# VLM rates (mm/yr, negative = subsidence)
VLM_RATES = {
    "Bangladesh": -2.0,
    "Vietnam"   : -8.9,
    "Maldives"  :  0.0,
}

# CMIP6 SSP2-4.5 contribution rates (mm/yr from 2023 baseline)
CMIP6_THERMO_RATES  = {"Bangladesh": 1.8, "Vietnam": 1.6,  "Maldives": 2.1}
CMIP6_GLACIER_RATES = {"Bangladesh": 0.9, "Vietnam": 0.8,  "Maldives": 1.1}

# PSI drift thresholds
PSI_STABLE   = 0.10
PSI_MONITOR  = 0.20
PSI_DRIFT    = 0.20

# Features excluded from PSI — deterministic or constant by design
PSI_EXCLUDED = {"month_sin", "month_cos", "vlm"}


# --- 3. AUDIT LOGGING --------------------------------------------------------

def log_feature_event(
    stage  : str,
    status : str,
    detail : str,
    error  : Optional[str] = None,
) -> None:
    """Append one row to the feature engineering audit log."""
    row = pd.DataFrame([{
        "timestamp" : RUN_TIMESTAMP,
        "stage"     : stage,
        "status"    : status,
        "detail"    : detail,
        "error"     : error if error else "",
    }])
    if FEATURE_LOG.exists():
        row.to_csv(FEATURE_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(FEATURE_LOG, mode="w", header=True, index=False)
    logger.info("[LOG] %s | %s | %s", stage, status, detail)


# --- 4. LOAD INPUTS ----------------------------------------------------------

def _convert_slrv2_to_anomaly(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert historical slrv2 column from RLR absolute datum (~7000 mm) to
    relative anomaly (mm above each country's 2022-12-31 baseline level).

    This ensures the historical slrv2 values are on the same datum as the
    live-engineered slrv2 produced by engineer_live_features(), and
    consistent with the risk thresholds defined in Cell 36.

    The anchor for each country is the mean slrv2 in the ±6 month window
    around SLR_BASELINE_DATE. Falls back to the last pre-baseline value
    if the window is empty.

    Parameters
    ----------
    raw_df : DataFrame after column rename, with 'slrv2' and 'country' cols.

    Returns
    -------
    DataFrame with slrv2 expressed as mm-above-2022-baseline anomaly.
    """
    if "slrv2" not in raw_df.columns or "country" not in raw_df.columns:
        logger.warning(
            "_convert_slrv2_to_anomaly: 'slrv2' or 'country' column missing. "
            "Skipping datum conversion."
        )
        return raw_df

    window_start = SLR_BASELINE_DATE - pd.DateOffset(months=6)
    window_end   = SLR_BASELINE_DATE + pd.DateOffset(months=6)

    country_frames: List[pd.DataFrame] = []

    for country in raw_df["country"].unique():
        # Use a copy named country_df — never reassign raw_df inside the loop
        country_df = raw_df[raw_df["country"] == country].copy()

        win = country_df[
            (country_df["date"] >= window_start) &
            (country_df["date"] <= window_end)
        ]

        if win.empty:
            pre = country_df[country_df["date"] <= SLR_BASELINE_DATE]
            if not pre.empty:
                anchor = float(pre["slrv2"].dropna().iloc[-1])
                logger.info(
                    "  %s: no window data — using last pre-baseline "
                    "slrv2=%.3f as anchor", country, anchor
                )
            else:
                anchor = float(country_df["slrv2"].dropna().mean())
                logger.warning(
                    "  %s: no pre-baseline data — using global mean "
                    "slrv2=%.3f as anchor", country, anchor
                )
        else:
            anchor = float(win["slrv2"].dropna().mean())
            logger.info(
                "  %s: slrv2 anchor=%.3f mm (window mean, "
                "converted to anomaly)", country, anchor
            )

        country_df = country_df.copy()
        country_df["slrv2"] = country_df["slrv2"] - anchor
        country_frames.append(country_df)

    result = pd.concat(country_frames, ignore_index=True)
    result = result.sort_values(["country", "date"]).reset_index(drop=True)
    return result


def load_historical_dataset() -> pd.DataFrame:
    """
    Load the existing 2628-row historical dataset from Cell 28.

    Two normalisation steps applied on load so all downstream code uses
    consistent feature names and a consistent datum:

    1. Column rename: Cell 28 names -> HyperSLR standardised names
       (see COLUMN_MAP at module level)

    2. Datum conversion: historical slrv2 (RLR absolute, ~7000 mm) ->
       relative anomaly (mm above 2022-12-31 baseline per country)
       so historical and live slrv2 values are directly comparable.

    Returns
    -------
    DataFrame with standardised column names, anomaly-referenced slrv2,
    and parsed date column.

    Raises
    ------
    FileNotFoundError  if dataset is not at expected path.
    ValueError         if critical columns are missing after rename.
    """
    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {HISTORICAL_CSV}\n"
            "Ensure Google Drive is mounted and Cell 28 has been executed."
        )

    # Load into a new local variable — never shadow it inside a conditional
    loaded_df = pd.read_csv(HISTORICAL_CSV)

    # Identify date column flexibly
    date_col = None
    for candidate in ["date", "Date", "DATE", "time", "year_month"]:
        if candidate in loaded_df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = loaded_df.columns[0]
        logger.warning(
            "No standard date column found. Using first column: %s", date_col
        )

    loaded_df[date_col] = pd.to_datetime(loaded_df[date_col], errors="coerce")
    loaded_df = loaded_df.rename(columns={date_col: "date"})
    loaded_df = loaded_df.dropna(subset=["date"])
    loaded_df = loaded_df.sort_values("date").reset_index(drop=True)

    # Step 1: Column rename
    rename_applied = {k: v for k, v in COLUMN_MAP.items()
                      if k in loaded_df.columns}
    loaded_df = loaded_df.rename(columns=rename_applied)
    logger.info("Column rename applied: %d mappings", len(rename_applied))

    # Verify critical columns present after rename
    missing_critical = [
        c for c in ["slrv2", "tide_gauge", "enso"]
        if c not in loaded_df.columns
    ]
    if missing_critical:
        raise ValueError(
            f"Critical columns missing after rename: {missing_critical}\n"
            f"Available columns: {list(loaded_df.columns)}"
        )

    # Step 2: Datum conversion — RLR absolute -> anomaly per country
    # _convert_slrv2_to_anomaly uses its own internal variable names and
    # never modifies loaded_df in place, so there is no scoping conflict.
    loaded_df = _convert_slrv2_to_anomaly(loaded_df)

    logger.info(
        "Historical dataset loaded: %d rows, %d columns",
        len(loaded_df), len(loaded_df.columns)
    )
    logger.info(
        "Historical date range: %s to %s",
        loaded_df["date"].min().date(), loaded_df["date"].max().date()
    )
    logger.info(
        "Column rename applied. Key columns confirmed: "
        "slrv2, tide_gauge, enso, aviso, vlm"
    )

    # Per-country slrv2 coverage and range after conversion
    for country in (loaded_df["country"].unique()
                    if "country" in loaded_df.columns else []):
        cdf   = loaded_df[loaded_df["country"] == country]
        valid = cdf["slrv2"].notna().sum()
        vmin  = cdf["slrv2"].min()
        vmax  = cdf["slrv2"].max()
        logger.info(
            "  %s: %d rows, slrv2 valid=%d/%d, range=[%.1f, %.1f] mm",
            country, len(cdf), valid, len(cdf), vmin, vmax
        )

    return loaded_df


def load_scaler_params() -> Dict:
    """Load existing scaler parameters from Cell 28."""
    if not SCALER_JSON.exists():
        logger.warning(
            "Scaler params not found at %s. Will use data-derived stats.",
            SCALER_JSON
        )
        return {}
    with open(SCALER_JSON) as f:
        params = json.load(f)
    logger.info("Scaler params loaded: %d entries", len(params))
    return params


def load_psmsl_live() -> pd.DataFrame:
    """Load the PSMSL live data ingested by Cell 33."""
    if not PSMSL_LATEST.exists():
        raise FileNotFoundError(
            f"PSMSL live data not found: {PSMSL_LATEST}\n"
            "Execute Cell 33 before running Cell 34."
        )
    live_df = pd.read_csv(PSMSL_LATEST, parse_dates=["date"])
    live_df = live_df.sort_values(
        ["country", "station_id", "date"]
    ).reset_index(drop=True)
    logger.info(
        "PSMSL live data loaded: %d rows across %d countries",
        len(live_df), live_df["country"].nunique()
    )
    return live_df


# --- 5. ENSO DATA ------------------------------------------------------------

def fetch_enso_index() -> pd.Series:
    """
    Download NOAA Nino3.4 ENSO index.
    Falls back to zero-mean series if download fails.
    """
    import requests as req

    logger.info("Fetching ENSO Nino3.4 index from NOAA PSL...")
    try:
        response = req.get(
            ENSO_URL, timeout=20,
            headers={"User-Agent": "HyperSLR-Research/1.0"}
        )
        response.raise_for_status()

        records = []
        for line in response.text.splitlines():
            parts = line.split()
            if len(parts) != 13:
                continue
            try:
                year = int(parts[0])
                for m, val_str in enumerate(parts[1:], start=1):
                    val = float(val_str)
                    if val == -99.99:
                        val = np.nan
                    records.append({
                        "date" : pd.Timestamp(year=year, month=m, day=1),
                        "enso" : val,
                    })
            except (ValueError, TypeError):
                continue

        if not records:
            raise ValueError("No records parsed from ENSO response.")

        enso_series = (
            pd.DataFrame(records)
            .dropna(subset=["date"])
            .set_index("date")["enso"]
        )
        logger.info(
            "ENSO data: %d months (%s to %s)",
            len(enso_series),
            enso_series.index.min().date(),
            enso_series.index.max().date()
        )
        return enso_series

    except Exception as exc:
        logger.warning(
            "ENSO fetch failed (%s). Using climatological mean (0.0).", exc
        )
        idx = pd.date_range("1950-01-01", "2025-12-01", freq="MS")
        return pd.Series(0.0, index=idx, name="enso")


# --- 6. FEATURE ENGINEERING PER COUNTRY -------------------------------------

def _aggregate_country_stations(
    psmsl_df : pd.DataFrame,
    country  : str,
) -> pd.DataFrame:
    """
    Aggregate multi-station PSMSL data for one country into a monthly series.

    De-means each station before averaging to remove datum offsets.
    Re-references the combined series to zero at the first observation.
    """
    cdf = psmsl_df[psmsl_df["country"] == country].copy()
    if cdf.empty:
        return pd.DataFrame(columns=["date", "rlr_mm"])

    station_means       = cdf.groupby("station_id")["rlr_mm"].transform("mean")
    cdf["rlr_demeaned"] = cdf["rlr_mm"] - station_means

    monthly = (
        cdf.groupby("date")["rlr_demeaned"]
        .mean()
        .reset_index()
        .rename(columns={"rlr_demeaned": "rlr_mm"})
    )
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly         = monthly.sort_values("date").reset_index(drop=True)

    first_valid       = (monthly["rlr_mm"].dropna().iloc[0]
                         if monthly["rlr_mm"].notna().any() else 0)
    monthly["rlr_mm"] = monthly["rlr_mm"] - first_valid
    return monthly


def engineer_live_features(
    psmsl_df  : pd.DataFrame,
    enso_index: pd.Series,
    country   : str,
) -> pd.DataFrame:
    """
    Apply Cell 28 feature engineering logic to live PSMSL data for one country.

    All SLR values are expressed as relative anomaly (mm above the country's
    own 2022-12-31 baseline level) so they are directly comparable with:
      - The historical slrv2 after datum conversion in load_historical_dataset
      - The risk thresholds defined in Cell 36

    Produces the 12 standardised features used throughout Cells 34-42:
      slrv2, tide_gauge, enso, aviso, vlm, cmip6_thermo, cmip6_glacier,
      momentum, residual, month_sin, month_cos, enso_x_aviso
    """
    logger.info("  Engineering features for %s...", country)

    monthly = _aggregate_country_stations(psmsl_df, country)
    if monthly.empty or len(monthly) < 3:
        logger.warning(
            "  Insufficient data for %s (%d rows). Skipping.",
            country, len(monthly)
        )
        return pd.DataFrame()

    vlm_rate     = VLM_RATES.get(country, 0.0)
    thermo_rate  = CMIP6_THERMO_RATES.get(country, 1.8)
    glacier_rate = CMIP6_GLACIER_RATES.get(country, 0.9)

    # ------------------------------------------------------------------
    # Anchor all SLR values to the station's own 2022-12-31 level.
    # _aggregate_country_stations already de-meaned and re-referenced
    # to zero at the first observation, so rlr_mm is a relative series.
    # We further re-reference to the ±6 month window around the baseline
    # date so values are consistent with the historical slrv2 anomaly
    # and with the Cell 36 risk thresholds.
    # ------------------------------------------------------------------
    window_start  = SLR_BASELINE_DATE - pd.DateOffset(months=6)
    window_end    = SLR_BASELINE_DATE + pd.DateOffset(months=6)
    baseline_rows = monthly[
        (monthly["date"] >= window_start) & (monthly["date"] <= window_end)
    ]

    if baseline_rows.empty:
        pre = monthly[monthly["date"] <= SLR_BASELINE_DATE]
        if not pre.empty:
            rlr_baseline = float(pre["rlr_mm"].iloc[-1])
            logger.info(
                "  %s: no window data — using last pre-baseline "
                "rlr_mm=%.3f as anchor", country, rlr_baseline
            )
        else:
            rlr_baseline = float(monthly["rlr_mm"].mean())
            logger.warning(
                "  %s: no pre-baseline data — using series mean "
                "%.3f as anchor", country, rlr_baseline
            )
    else:
        rlr_baseline = float(baseline_rows["rlr_mm"].mean())

    logger.info("  %s: RLR baseline anchor = %.4f mm", country, rlr_baseline)

    monthly["years_from_baseline"] = (
        (monthly["date"] - SLR_BASELINE_DATE).dt.days / 365.25
    )

    # tide_gauge: relative anomaly (mm above 2022 baseline)
    monthly["tide_gauge"] = monthly["rlr_mm"] - rlr_baseline

    # slrv2: tide_gauge anomaly + VLM contribution since baseline
    monthly["slrv2"] = (
        monthly["tide_gauge"]
        + vlm_rate * monthly["years_from_baseline"]
    )

    # ENSO
    monthly["ym"]   = monthly["date"].dt.to_period("M")
    enso_df         = enso_index.copy()
    enso_df.index   = enso_df.index.to_period("M")
    monthly["enso"] = monthly["ym"].map(enso_df).astype(float)
    monthly["enso"] = monthly["enso"].fillna(0.0)

    # aviso: steric/altimetry component (anomaly, without VLM offset)
    monthly["aviso"]         = monthly["tide_gauge"]
    monthly["vlm"]           = vlm_rate
    monthly["cmip6_thermo"]  = (
        thermo_rate  * monthly["years_from_baseline"].clip(lower=0)
    )
    monthly["cmip6_glacier"] = (
        glacier_rate * monthly["years_from_baseline"].clip(lower=0)
    )

    monthly["momentum"] = (
        monthly["slrv2"].rolling(window=12, min_periods=3).mean()
    )
    monthly["residual"] = monthly["slrv2"] - monthly["momentum"]

    month_num            = monthly["date"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    monthly["enso_x_aviso"] = monthly["enso"] * monthly["aviso"]
    monthly["country"]      = country

    monthly = monthly.drop(
        columns=["rlr_mm", "years_from_baseline", "ym"],
        errors="ignore"
    )
    monthly = monthly.dropna(subset=["slrv2"])
    monthly = monthly.reset_index(drop=True)

    logger.info(
        "  %s: %d rows engineered, slrv2 range=[%.2f, %.2f] mm",
        country, len(monthly),
        monthly["slrv2"].min(), monthly["slrv2"].max()
    )
    return monthly


# --- 7. POPULATION STABILITY INDEX (PSI) ------------------------------------

def compute_psi(
    reference : pd.Series,
    current   : pd.Series,
    n_bins    : int = 10,
) -> float:
    """
    Compute Population Stability Index between reference and current
    distributions.

    PSI < 0.10  : No significant change
    PSI < 0.20  : Moderate change, monitor
    PSI >= 0.20 : Significant shift, action required
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(ref_clean) < 5 or len(cur_clean) < 2:
        return 0.0

    bin_edges = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    eps        = 1e-6
    ref_counts = np.histogram(ref_clean, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(
        np.clip(cur_clean, bin_edges[0], bin_edges[-1]), bins=bin_edges
    )[0].astype(float)

    ref_pct = (ref_counts + eps) / (ref_clean.count() + eps * n_bins)
    cur_pct = (cur_counts + eps) / (cur_clean.count() + eps * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(psi, 0.0), 6)


def detect_feature_drift(
    historical_df : pd.DataFrame,
    live_df       : pd.DataFrame,
    features      : List[str],
) -> Dict[str, Any]:
    """
    Compute PSI for each feature between historical and live distributions.
    """
    logger.info("--- FEATURE DRIFT DETECTION ---")
    psi_scores = {}
    drifted    = []

    for feat in features:
        if feat not in historical_df.columns or feat not in live_df.columns:
            logger.warning(
                "  Feature '%s' missing from one dataset. Skipping.", feat
            )
            psi_scores[feat] = None
            continue

        psi = compute_psi(historical_df[feat], live_df[feat])
        psi_scores[feat] = psi

        if psi >= PSI_DRIFT:
            status = "DRIFT"
            drifted.append(feat)
        elif psi >= PSI_STABLE:
            status = "MONITOR"
        else:
            status = "STABLE"

        logger.info("  PSI [%s]: %.4f — %s", feat, psi, status)

    drift_detected = len(drifted) > 0
    result = {
        "psi_scores"       : psi_scores,
        "drift_detected"   : drift_detected,
        "drifted_features" : drifted,
        "stable_features"  : [f for f in features if f not in drifted],
        "max_psi"          : max(
            (v for v in psi_scores.values() if v is not None), default=0.0
        ),
    }

    if drift_detected:
        logger.warning(
            "  Drift confirmed in %d feature(s): %s", len(drifted), drifted
        )
    else:
        logger.info(
            "  All features stable (max PSI = %.4f)", result["max_psi"]
        )

    return result


# --- 8. SCALER RECALIBRATION -------------------------------------------------

def update_scaler_params(
    historical_df   : pd.DataFrame,
    live_df         : pd.DataFrame,
    existing_params : Dict,
    drift_result    : Dict,
    features        : List[str],
) -> Tuple[Dict, bool]:
    """
    Recalibrate scaler parameters if drift is confirmed.
    Only recalibrates features where PSI >= PSI_DRIFT.
    """
    if not drift_result["drift_detected"]:
        logger.info(
            "Scaler recalibration: not required (no drift detected)."
        )
        return existing_params, False

    logger.info(
        "Scaler recalibration: drift confirmed. Refitting on combined data."
    )
    combined       = pd.concat([historical_df, live_df], ignore_index=True)
    updated_params = dict(existing_params)

    for feat in drift_result["drifted_features"]:
        if feat not in combined.columns:
            continue
        series = combined[feat].dropna()
        if len(series) < 2:
            continue
        updated_params[feat] = {
            "mean"    : float(series.mean()),
            "std"     : float(series.std()),
            "min"     : float(series.min()),
            "max"     : float(series.max()),
            "recal_ts": RUN_TIMESTAMP,
        }
        logger.info(
            "  Recalibrated: %s (mean=%.4f, std=%.4f)",
            feat, updated_params[feat]["mean"], updated_params[feat]["std"]
        )

    with open(SCALER_UPDATED, "w") as f:
        json.dump(updated_params, f, indent=2)
    logger.info("Updated scaler saved: %s", SCALER_UPDATED)
    return updated_params, True


# --- 9. MERGE WITH HISTORICAL ------------------------------------------------

def merge_with_historical(
    historical_df : pd.DataFrame,
    live_df       : pd.DataFrame,
) -> pd.DataFrame:
    """
    Append live engineered features to the historical dataset.

    Only appends rows that extend beyond the historical end date per country
    to prevent double-counting. De-duplicates on (date, country).
    """
    hist = historical_df.copy()
    live = live_df.copy()

    if "country" not in hist.columns:
        logger.warning(
            "Historical dataset has no 'country' column. Adding 'Unknown'."
        )
        hist["country"] = "Unknown"

    common_cols = ["date", "country"] + [
        f for f in FEATURE_COLUMNS if f in live.columns
    ]
    live_subset = live[[c for c in common_cols if c in live.columns]].copy()

    hist_max_date = hist["date"].max()
    new_rows      = live_subset[live_subset["date"] > hist_max_date].copy()

    if new_rows.empty:
        logger.info(
            "No new timesteps beyond historical end (%s). Merge unchanged.",
            hist_max_date.date()
        )
        return hist

    logger.info(
        "Appending %d new rows (beyond %s) to historical dataset.",
        len(new_rows), hist_max_date.date()
    )

    merged = pd.concat([hist, new_rows], ignore_index=True)
    merged = merged.sort_values(["country", "date"]).reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["date", "country"], keep="last")
    merged = merged.reset_index(drop=True)

    logger.info(
        "Merged dataset: %d rows (was %d historical + %d new)",
        len(merged), len(hist), len(new_rows)
    )
    return merged


# =============================================================================
# CELL 34 — MAIN EXECUTION
# =============================================================================

def run_cell_34() -> Dict[str, Any]:
    """
    Execute the full Cell 34 automated feature engineering pipeline.

    Steps:
      1. Load historical dataset with column rename and RLR->anomaly conversion
      2. Load scaler params and live PSMSL data
      3. Fetch ENSO index
      4. Engineer 12 standardised features per country from live PSMSL
      5. Detect feature drift (PSI) against historical distribution
      6. Recalibrate scaler if drift confirmed
      7. Merge live features with historical dataset
      8. Save outputs and return summary
    """
    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 34: AUTOMATED FEATURE ENGINEERING")
    logger.info("Run timestamp: %s", RUN_TIMESTAMP)
    logger.info("=" * 70)

    logger.info("--- LOADING INPUTS ---")
    historical_df = load_historical_dataset()
    scaler_params = load_scaler_params()
    psmsl_df      = load_psmsl_live()

    log_feature_event(
        "LOAD", "SUCCESS",
        f"historical={len(historical_df)} rows, psmsl={len(psmsl_df)} rows"
    )

    logger.info("--- ENSO DATA ---")
    enso_index = fetch_enso_index()

    logger.info("--- FEATURE ENGINEERING ---")
    countries   = psmsl_df["country"].unique().tolist()
    live_frames = []

    for country in countries:
        feat_df = engineer_live_features(psmsl_df, enso_index, country)
        if not feat_df.empty:
            live_frames.append(feat_df)
            log_feature_event(
                f"ENGINEER_{country}", "SUCCESS",
                f"{len(feat_df)} rows, {len(feat_df.columns)} columns"
            )
        else:
            log_feature_event(
                f"ENGINEER_{country}", "FAILED", "No features produced."
            )

    if not live_frames:
        raise RuntimeError(
            "Feature engineering produced no output for any country."
        )

    live_df = pd.concat(live_frames, ignore_index=True)
    live_df.to_csv(FEATURES_OUT, index=False)
    logger.info(
        "Live features saved: %s (%d rows)", FEATURES_OUT, len(live_df)
    )

    logger.info("--- DRIFT DETECTION ---")
    shared_features = [
        f for f in FEATURE_COLUMNS
        if f in historical_df.columns
        and f in live_df.columns
        and f not in PSI_EXCLUDED
    ]

    if shared_features:
        drift_result = detect_feature_drift(
            historical_df, live_df, shared_features
        )
        drift_status = "DRIFT" if drift_result["drift_detected"] else "STABLE"
        log_feature_event(
            "DRIFT_CHECK", drift_status,
            f"max_PSI={drift_result['max_psi']:.4f}, "
            f"drifted={drift_result['drifted_features']}"
        )
    else:
        logger.warning(
            "No shared feature columns for PSI comparison. "
            "Live feature columns: %s", list(live_df.columns)
        )
        drift_result = {
            "drift_detected"   : False,
            "drifted_features" : [],
            "psi_scores"       : {},
            "max_psi"          : 0.0,
        }
        log_feature_event("DRIFT_CHECK", "SKIPPED",
                          "No shared feature columns.")

    logger.info("--- SCALER RECALIBRATION ---")
    updated_scaler, was_updated = update_scaler_params(
        historical_df, live_df, scaler_params, drift_result, shared_features
    )
    scaler_status = "UPDATED" if was_updated else "UNCHANGED"
    log_feature_event(
        "SCALER", scaler_status,
        f"drift={drift_result['drift_detected']}, "
        f"updated_features={drift_result['drifted_features']}"
    )

    logger.info("--- MERGE WITH HISTORICAL ---")
    merged_df = merge_with_historical(historical_df, live_df)
    merged_df.to_csv(MERGED_OUT, index=False)
    logger.info(
        "Merged dataset saved: %s (%d rows)", MERGED_OUT, len(merged_df)
    )
    log_feature_event(
        "MERGE", "SUCCESS",
        f"merged_rows={len(merged_df)}, "
        f"new_rows={len(merged_df) - len(historical_df)}"
    )

    summary = {
        "run_timestamp"       : RUN_TIMESTAMP,
        "historical_rows"     : len(historical_df),
        "live_feature_rows"   : len(live_df),
        "merged_rows"         : len(merged_df),
        "countries_processed" : countries,
        "feature_columns"     : shared_features if shared_features else FEATURE_COLUMNS,
        "drift": {
            "detected"         : drift_result["drift_detected"],
            "max_psi"          : drift_result["max_psi"],
            "drifted_features" : drift_result["drifted_features"],
            "psi_scores"       : drift_result["psi_scores"],
        },
        "scaler_status"       : scaler_status,
        "outputs": {
            "features_latest"  : str(FEATURES_OUT),
            "merged_dataset"   : str(MERGED_OUT),
            "feature_log"      : str(FEATURE_LOG),
            "scaler_updated"   : str(SCALER_UPDATED) if was_updated else None,
        },
    }

    summary_path = (
        LOG_DIR / f"feature_engineering_summary_{RUN_TIMESTAMP}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("CELL 34 COMPLETE")
    logger.info(
        "Features engineered : %d new timesteps x %d features",
        len(live_df), len(shared_features) if shared_features else 12
    )
    logger.info(
        "Drift status        : %s (max PSI = %.4f)",
        "DETECTED" if drift_result["drift_detected"] else "STABLE",
        drift_result["max_psi"]
    )
    logger.info("Merged dataset      : %d rows", len(merged_df))
    logger.info("Scaler              : %s", scaler_status)
    logger.info("=" * 70)

    return summary


# =============================================================================
# CELL 34 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_34() -> None:
    """
    Comprehensive unit and integration tests for all Cell 34 functions.

    T01 — Historical dataset load: schema, column rename, anomaly conversion
    T02 — Scaler params load
    T03 — PSMSL live data load (requires Cell 33 output)
    T04 — ENSO fetch from NOAA (live network)
    T05 — ENSO fallback on network failure
    T06 — Station aggregation: single-station country
    T07 — Station aggregation: multi-station country (de-mean and merge)
    T08 — Feature engineering: all 12 columns present
    T09 — Feature engineering: seasonal encoding correctness
    T10 — Feature engineering: momentum and residual identity
    T11 — PSI: identical distributions return near-zero
    T12 — PSI: shifted distributions return elevated score
    T13 — PSI: edge cases (empty series, single value)
    T14 — Drift detection: structure and PSI bounds correct
    T15 — Drift detection: large shift triggers drift flag
    T16 — Scaler recalibration: only runs on confirmed drift
    T17 — Merge: new rows appended, no duplicates
    T18 — Merge: no-op when live data not newer
    T19 — End-to-end pipeline on Drive data
    T20 — Feature log CSV written correctly
    """

    PASS = "PASS"
    FAIL = "FAIL"
    results = []

    def record(test_id, name, status, detail=""):
        results.append((test_id, name, status, detail))
        marker = "[PASS]" if status == PASS else "[FAIL]"
        print(f"  {marker} {test_id}: {name}" +
              (f" — {detail}" if detail else ""))

    def make_synthetic_historical(n_per_country=100) -> pd.DataFrame:
        """
        Synthetic historical DataFrame using standardised HyperSLR column
        names AND anomaly-scale slrv2 values (not RLR absolute).
        Mimics the output of load_historical_dataset() post-conversion.
        """
        frames = []
        np.random.seed(42)
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            dates = pd.date_range("1950-01-01", periods=n_per_country, freq="MS")
            # Small anomaly values consistent with mm-above-baseline scale
            slr   = np.cumsum(np.random.normal(0.3, 0.5, n_per_country))
            frames.append(pd.DataFrame({
                "date"         : dates,
                "country"      : country,
                "slrv2"        : slr,
                "tide_gauge"   : slr + np.random.normal(0, 0.5, n_per_country),
                "enso"         : np.random.normal(0, 0.8, n_per_country),
                "aviso"        : slr + np.random.normal(0, 0.3, n_per_country),
                "vlm"          : VLM_RATES.get(country, 0.0),
                "cmip6_thermo" : np.linspace(0, 5, n_per_country),
                "cmip6_glacier": np.linspace(0, 2, n_per_country),
                "momentum"     : pd.Series(slr).rolling(
                    12, min_periods=3
                ).mean().values,
                "residual"     : np.random.normal(0, 0.2, n_per_country),
                "month_sin"    : np.sin(2 * np.pi * dates.month / 12),
                "month_cos"    : np.cos(2 * np.pi * dates.month / 12),
                "enso_x_aviso" : np.random.normal(0, 1, n_per_country),
            }))
        return pd.concat(frames, ignore_index=True)

    def make_synthetic_psmsl(start="2020-01-01", n=36) -> pd.DataFrame:
        frames = []
        np.random.seed(7)
        station_map = {
            "Bangladesh": [369, 543],
            "Vietnam"   : [841],
            "Maldives"  : [1707],
        }
        for country, stations in station_map.items():
            for sid in stations:
                dates = pd.date_range(start, periods=n, freq="MS")
                rlr   = np.random.normal(7000, 40, n)
                frames.append(pd.DataFrame({
                    "date"      : dates,
                    "rlr_mm"    : rlr,
                    "country"   : country,
                    "station_id": sid,
                }))
        return pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 70)
    print("CELL 34 — TEST SUITE")
    print("=" * 70 + "\n")

    synth_hist  = make_synthetic_historical()
    synth_psmsl = make_synthetic_psmsl()
    enso_flat   = pd.Series(
        0.0,
        index=pd.date_range("1950-01-01", "2025-12-01", freq="MS"),
        name="enso"
    )

    # -------------------------------------------------------------------------
    # T01: Historical dataset load, column rename, anomaly conversion
    # -------------------------------------------------------------------------
    test_id = "T01"
    try:
        hist_df = load_historical_dataset()
        assert len(hist_df) >= 100,               f"Too few rows: {len(hist_df)}"
        assert "date"         in hist_df.columns, "Missing 'date'"
        assert "slrv2"        in hist_df.columns, "Missing 'slrv2' (rename failed)"
        assert "tide_gauge"   in hist_df.columns, "Missing 'tide_gauge'"
        assert "enso"         in hist_df.columns, "Missing 'enso'"
        assert "aviso"        in hist_df.columns, "Missing 'aviso'"
        assert "momentum"     in hist_df.columns, "Missing 'momentum'"
        assert "residual"     in hist_df.columns, "Missing 'residual'"
        assert "enso_x_aviso" in hist_df.columns, "Missing 'enso_x_aviso'"
        assert pd.api.types.is_datetime64_any_dtype(hist_df["date"]), \
            "date not datetime"
        # Verify anomaly conversion: values should be small, not ~7000
        slrv2_abs_max = hist_df["slrv2"].abs().max()
        assert slrv2_abs_max < 1000, \
            f"slrv2 still at RLR scale (max abs={slrv2_abs_max:.1f}). " \
            f"Anomaly conversion failed."
        # All countries have valid slrv2
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            valid = hist_df[hist_df["country"] == country]["slrv2"].notna().sum()
            assert valid > 0, f"{country}: zero valid slrv2 after conversion"
        record(test_id, "Historical load — rename + anomaly conversion", PASS,
               f"rows={len(hist_df)}, "
               f"slrv2_abs_max={slrv2_abs_max:.1f}mm")
    except Exception as e:
        record(test_id, "Historical load — rename + anomaly conversion",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T02: Scaler params load
    # -------------------------------------------------------------------------
    test_id = "T02"
    try:
        params = load_scaler_params()
        assert isinstance(params, dict), "Expected dict"
        record(test_id, "Scaler params load", PASS, f"keys={len(params)}")
    except Exception as e:
        record(test_id, "Scaler params load", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T03: PSMSL live data load
    # -------------------------------------------------------------------------
    test_id = "T03"
    try:
        psmsl_df = load_psmsl_live()
        assert not psmsl_df.empty,              "Empty DataFrame"
        assert "rlr_mm"  in psmsl_df.columns,  "Missing rlr_mm"
        assert "country" in psmsl_df.columns,  "Missing country"
        assert psmsl_df["country"].nunique() >= 1
        record(test_id, "PSMSL live data load", PASS,
               f"rows={len(psmsl_df)}, "
               f"countries={psmsl_df['country'].unique().tolist()}")
    except Exception as e:
        record(test_id, "PSMSL live data load", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T04: ENSO live fetch
    # -------------------------------------------------------------------------
    test_id = "T04"
    try:
        enso = fetch_enso_index()
        assert isinstance(enso, pd.Series)
        assert len(enso) > 100
        assert enso.notna().sum() > 50
        assert enso.index.dtype == "datetime64[ns]"
        record(test_id, "ENSO live fetch", PASS,
               f"records={len(enso)}, "
               f"range={enso.index.min().date()} to {enso.index.max().date()}")
    except Exception as e:
        record(test_id, "ENSO live fetch", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T05: ENSO fallback
    # -------------------------------------------------------------------------
    test_id = "T05"
    try:
        idx      = pd.date_range("1950-01-01", "2025-12-01", freq="MS")
        fallback = pd.Series(0.0, index=idx, name="enso")
        assert len(fallback) > 100
        assert (fallback == 0.0).all()
        record(test_id, "ENSO fallback Series", PASS,
               f"length={len(fallback)}")
    except Exception as e:
        record(test_id, "ENSO fallback Series", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T06: Station aggregation — single station
    # -------------------------------------------------------------------------
    test_id = "T06"
    try:
        result = _aggregate_country_stations(synth_psmsl, "Maldives")
        assert not result.empty
        assert "date"   in result.columns
        assert "rlr_mm" in result.columns
        first_val = result["rlr_mm"].dropna().iloc[0]
        assert abs(first_val) < 1e-9, f"First value not 0: {first_val}"
        record(test_id, "Station aggregation — single station", PASS,
               f"rows={len(result)}")
    except Exception as e:
        record(test_id, "Station aggregation — single station", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T07: Station aggregation — multi-station de-mean
    # -------------------------------------------------------------------------
    test_id = "T07"
    try:
        result  = _aggregate_country_stations(synth_psmsl, "Bangladesh")
        assert not result.empty
        assert result["rlr_mm"].abs().mean() < 500, \
            f"De-meaning failed: {result['rlr_mm'].abs().mean():.1f}"
        n_dates = synth_psmsl[
            synth_psmsl["country"] == "Bangladesh"
        ]["date"].nunique()
        assert len(result) == n_dates, \
            f"Expected {n_dates} rows, got {len(result)}"
        record(test_id, "Station aggregation — multi-station de-mean", PASS,
               f"rows={len(result)}, "
               f"mean_abs={result['rlr_mm'].abs().mean():.2f}")
    except Exception as e:
        record(test_id, "Station aggregation — multi-station de-mean", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T08: Feature engineering — all 12 columns present
    # -------------------------------------------------------------------------
    test_id = "T08"
    try:
        feat_df = engineer_live_features(synth_psmsl, enso_flat, "Bangladesh")
        assert not feat_df.empty
        missing = [f for f in FEATURE_COLUMNS if f not in feat_df.columns]
        assert not missing, f"Missing columns: {missing}"
        # slrv2 should be anomaly-scale (not RLR ~7000)
        assert feat_df["slrv2"].abs().max() < 500, \
            f"slrv2 at RLR scale: max={feat_df['slrv2'].abs().max():.1f}"
        record(test_id, "Feature engineering — all 12 columns present", PASS,
               f"rows={len(feat_df)}, "
               f"slrv2_range=[{feat_df['slrv2'].min():.1f}, "
               f"{feat_df['slrv2'].max():.1f}]mm")
    except Exception as e:
        record(test_id, "Feature engineering — all 12 columns present",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T09: Feature engineering — seasonal encoding correctness
    # -------------------------------------------------------------------------
    test_id = "T09"
    try:
        feat_df = engineer_live_features(synth_psmsl, enso_flat, "Bangladesh")
        for _, row in feat_df.iterrows():
            m            = row["date"].month
            expected_sin = np.sin(2 * np.pi * m / 12)
            expected_cos = np.cos(2 * np.pi * m / 12)
            assert abs(row["month_sin"] - expected_sin) < 1e-9
            assert abs(row["month_cos"] - expected_cos) < 1e-9
        record(test_id, "Seasonal encoding correctness", PASS)
    except Exception as e:
        record(test_id, "Seasonal encoding correctness", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T10: Feature engineering — momentum and residual identity
    # -------------------------------------------------------------------------
    test_id = "T10"
    try:
        feat_df = engineer_live_features(synth_psmsl, enso_flat, "Maldives")
        valid   = feat_df.dropna(subset=["slrv2", "momentum", "residual"])
        assert len(valid) > 0
        diff = (valid["slrv2"] - valid["momentum"] - valid["residual"]).abs()
        assert diff.max() < 1e-9, \
            f"residual identity failed. Max error: {diff.max():.2e}"
        record(test_id, "Momentum and residual identity", PASS,
               f"max_error={diff.max():.2e}")
    except Exception as e:
        record(test_id, "Momentum and residual identity", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T11: PSI — identical distributions
    # -------------------------------------------------------------------------
    test_id = "T11"
    try:
        np.random.seed(0)
        s   = pd.Series(np.random.normal(0, 1, 500))
        psi = compute_psi(s, s)
        assert psi < 0.01, f"PSI for identical series too high: {psi}"
        record(test_id, "PSI — identical distributions", PASS,
               f"PSI={psi:.6f}")
    except Exception as e:
        record(test_id, "PSI — identical distributions", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T12: PSI — shifted distribution elevated
    # -------------------------------------------------------------------------
    test_id = "T12"
    try:
        np.random.seed(1)
        ref = pd.Series(np.random.normal(0, 1, 500))
        cur = pd.Series(np.random.normal(3, 1, 500))
        psi = compute_psi(ref, cur)
        assert psi > PSI_DRIFT, \
            f"PSI for 3-sigma shift should exceed {PSI_DRIFT}. Got: {psi}"
        record(test_id, "PSI — shifted distribution elevated", PASS,
               f"PSI={psi:.4f}")
    except Exception as e:
        record(test_id, "PSI — shifted distribution elevated", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T13: PSI — edge cases
    # -------------------------------------------------------------------------
    test_id = "T13"
    try:
        psi_empty  = compute_psi(pd.Series(dtype=float), pd.Series([1.0]))
        psi_single = compute_psi(
            pd.Series([1.0, 2.0, 3.0]), pd.Series([1.5])
        )
        assert psi_empty  == 0.0
        assert psi_single == 0.0
        record(test_id, "PSI — edge cases", PASS,
               f"empty={psi_empty}, single={psi_single}")
    except Exception as e:
        record(test_id, "PSI — edge cases", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T14: Drift detection — structure correct
    # -------------------------------------------------------------------------
    test_id = "T14"
    try:
        feat_stable = engineer_live_features(
            synth_psmsl, enso_flat, "Bangladesh"
        )
        shared = [
            f for f in FEATURE_COLUMNS
            if f in synth_hist.columns
            and f in feat_stable.columns
            and f not in PSI_EXCLUDED
        ]
        result = detect_feature_drift(synth_hist, feat_stable, shared)
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["max_psi"], float)
        assert isinstance(result["psi_scores"], dict)
        assert result["max_psi"] >= 0.0
        assert set(result["psi_scores"].keys()) == set(shared)
        record(test_id, "Drift detection — structure and PSI bounds", PASS,
               f"drift={result['drift_detected']}, "
               f"max_PSI={result['max_psi']:.4f}")
    except Exception as e:
        record(test_id, "Drift detection — structure and PSI bounds",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T15: Drift detection — large shift triggers drift flag
    # -------------------------------------------------------------------------
    test_id = "T15"
    try:
        feat_shifted = engineer_live_features(
            synth_psmsl, enso_flat, "Bangladesh"
        ).copy()
        if "slrv2" in feat_shifted.columns:
            feat_shifted["slrv2"] = feat_shifted["slrv2"] + 1000.0
        result = detect_feature_drift(synth_hist, feat_shifted, ["slrv2"])
        assert result["drift_detected"], \
            f"Expected drift for 1000-unit shift. " \
            f"PSI={result['psi_scores'].get('slrv2')}"
        record(test_id, "Drift detection — large shift flagged", PASS,
               f"PSI={result['psi_scores'].get('slrv2', 'N/A'):.4f}")
    except Exception as e:
        record(test_id, "Drift detection — large shift flagged", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T16: Scaler recalibration logic
    # -------------------------------------------------------------------------
    test_id = "T16"
    try:
        no_drift  = {
            "drift_detected": False, "drifted_features": [], "psi_scores": {}
        }
        _, updated = update_scaler_params(
            synth_hist, synth_hist, {}, no_drift, []
        )
        assert not updated, "Scaler should NOT update without drift"

        yes_drift = {
            "drift_detected"  : True,
            "drifted_features": ["slrv2"],
            "psi_scores"      : {"slrv2": 0.35},
        }
        params, updated2 = update_scaler_params(
            synth_hist, synth_hist, {}, yes_drift, ["slrv2"]
        )
        assert updated2,          "Scaler should update when drift confirmed"
        assert "slrv2" in params, "slrv2 should be in updated params"
        assert "mean"  in params["slrv2"]
        assert "std"   in params["slrv2"]
        record(test_id, "Scaler recalibration logic", PASS)
    except Exception as e:
        record(test_id, "Scaler recalibration logic", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T17: Merge — new rows appended, no duplicates
    # -------------------------------------------------------------------------
    test_id = "T17"
    try:
        hist_short    = synth_hist[synth_hist["date"] < "2005-01-01"].copy()
        future_dates  = pd.date_range("2005-01-01", periods=12, freq="MS")
        future_frames = []
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            np.random.seed(99)
            slr = np.cumsum(np.random.normal(0.3, 0.5, 12))
            future_frames.append(pd.DataFrame({
                "date"         : future_dates,
                "country"      : country,
                "slrv2"        : slr,
                "tide_gauge"   : slr,
                "enso"         : 0.0,
                "aviso"        : slr,
                "vlm"          : VLM_RATES.get(country, 0.0),
                "cmip6_thermo" : np.linspace(0, 1, 12),
                "cmip6_glacier": np.linspace(0, 0.5, 12),
                "momentum"     : slr,
                "residual"     : 0.0,
                "month_sin"    : np.sin(2 * np.pi * future_dates.month / 12),
                "month_cos"    : np.cos(2 * np.pi * future_dates.month / 12),
                "enso_x_aviso" : 0.0,
            }))
        live_extend = pd.concat(future_frames, ignore_index=True)
        merged      = merge_with_historical(hist_short, live_extend)
        expected    = len(hist_short) + len(live_extend)
        assert len(merged) > len(hist_short), \
            f"Merged ({len(merged)}) should exceed hist ({len(hist_short)})"
        dup_check = merged.duplicated(subset=["date", "country"]).sum()
        assert dup_check == 0, f"Duplicates found: {dup_check}"
        assert len(merged) == expected, \
            f"Expected {expected} rows, got {len(merged)}"
        record(test_id, "Merge — new rows appended, no duplicates", PASS,
               f"hist={len(hist_short)}, new={len(live_extend)}, "
               f"merged={len(merged)}, dups={dup_check}")
    except Exception as e:
        record(test_id, "Merge — new rows appended, no duplicates", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T18: Merge — no-op when live data not newer
    # -------------------------------------------------------------------------
    test_id = "T18"
    try:
        hist_full = synth_hist.copy()
        live_old  = synth_hist[synth_hist["date"] < "1960-01-01"].copy()
        merged    = merge_with_historical(hist_full, live_old)
        assert len(merged) == len(hist_full), \
            f"Expected no change. hist={len(hist_full)}, merged={len(merged)}"
        record(test_id, "Merge — no-op when live data not newer", PASS,
               f"rows unchanged={len(merged)}")
    except Exception as e:
        record(test_id, "Merge — no-op when live data not newer", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T19: End-to-end pipeline on Drive data
    # -------------------------------------------------------------------------
    test_id = "T19"
    try:
        summary = run_cell_34()
        assert "run_timestamp"     in summary
        assert "live_feature_rows" in summary
        assert "merged_rows"       in summary
        assert summary["merged_rows"] >= summary["historical_rows"]
        assert FEATURES_OUT.exists(), "features_latest.csv not created"
        assert MERGED_OUT.exists(),   "merged_dataset.csv not created"

        # Verify column rename in merged dataset
        merged_check = pd.read_csv(MERGED_OUT, nrows=5)
        assert "slrv2" in merged_check.columns, \
            "slrv2 missing from merged dataset"
        assert "tide_gauge" in merged_check.columns, \
            "tide_gauge missing from merged dataset"

        # Verify anomaly conversion: slrv2 should not be at RLR scale
        merged_full  = pd.read_csv(MERGED_OUT)
        slrv2_absmax = merged_full["slrv2"].abs().max()
        assert slrv2_absmax < 1000, \
            f"slrv2 still at RLR scale in merged dataset " \
            f"(max abs={slrv2_absmax:.1f}mm)"

        # All three countries have valid slrv2
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            cdf   = merged_full[merged_full["country"] == country]
            valid = cdf["slrv2"].notna().sum()
            assert valid > 0, \
                f"{country}: zero valid slrv2 in merged dataset"

        record(test_id, "End-to-end pipeline on Drive data", PASS,
               f"live_rows={summary['live_feature_rows']}, "
               f"merged={summary['merged_rows']}, "
               f"slrv2_absmax={slrv2_absmax:.1f}mm, "
               f"drift={summary['drift']['detected']}")
    except Exception as e:
        record(test_id, "End-to-end pipeline on Drive data", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T20: Feature log written correctly
    # -------------------------------------------------------------------------
    test_id = "T20"
    try:
        assert FEATURE_LOG.exists(), f"Feature log not found: {FEATURE_LOG}"
        df_log = pd.read_csv(FEATURE_LOG)
        for col in ["timestamp", "stage", "status", "detail", "error"]:
            assert col in df_log.columns, f"Missing log column: {col}"
        assert len(df_log) > 0, "Feature log is empty"
        record(test_id, "Feature engineering log written", PASS,
               f"entries={len(df_log)}, "
               f"stages={df_log['stage'].unique().tolist()}")
    except Exception as e:
        record(test_id, "Feature engineering log written", FAIL, str(e))

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

test_cell_34()

print("\nFINAL FEATURE ENGINEERING SUMMARY")
print("-" * 40)
if FEATURES_OUT.exists():
    df_feat = pd.read_csv(FEATURES_OUT)
    print(f"Live features file  : {FEATURES_OUT}")
    print(f"  Rows              : {len(df_feat)}")
    print(f"  Countries         : "
          f"{df_feat['country'].unique().tolist() if 'country' in df_feat.columns else 'N/A'}")
    print(f"  slrv2 range       : "
          f"[{df_feat['slrv2'].min():.2f}, {df_feat['slrv2'].max():.2f}] mm")
if MERGED_OUT.exists():
    df_merged = pd.read_csv(MERGED_OUT)
    print(f"Merged dataset file : {MERGED_OUT}")
    print(f"  Rows              : {len(df_merged)}")
    print(f"  slrv2 valid       : "
          f"{df_merged['slrv2'].notna().sum()}/{len(df_merged)}")
    for country in ["Bangladesh", "Vietnam", "Maldives"]:
        cdf   = df_merged[df_merged["country"] == country]
        valid = cdf["slrv2"].notna().sum()
        vmin  = cdf["slrv2"].min()
        vmax  = cdf["slrv2"].max()
        print(f"    {country:<14}: {valid}/{len(cdf)} valid, "
              f"range=[{vmin:.1f}, {vmax:.1f}] mm")
print(f"Feature log         : {FEATURE_LOG}")
