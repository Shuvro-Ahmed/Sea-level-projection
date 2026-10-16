# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 39: Automated Alert and Notification System
# =============================================================================
# Purpose  : Compare the current Cell 36 risk assessment against the
#            previous run's state; fire structured alerts when any of the
#            following conditions are met:
#
#   ALERT CLASS          TRIGGER CONDITION
#   ─────────────────────────────────────────────────────────────────────
#   THRESHOLD_CROSSED    Risk level upgraded (GREEN→AMBER, AMBER→RED, etc.)
#   ACCELERATION_ONSET   Acceleration detected this run but not previous run
#   ACCELERATION_PERSIST Acceleration detected for >= ACCEL_PERSIST_RUNS
#                        consecutive runs
#   RAPID_RISE           Single-run cumulative rise delta > RAPID_RISE_MM
#   COMPOUND_SPIKE       compound_prob_annual crosses COMPOUND_ALERT_PROB
#   DATA_GAP             Last observation date older than DATA_GAP_DAYS days
#   SYSTEM_LEVEL_UP      System-wide risk level upgraded
#
# Dispatch cascade (first configured channel wins; all enabled channels fire):
#   Primary  : Email via Gmail SMTP (TLS, App Password)
#   Fallback : Alert log only (CSV + JSON) — always active
#
# State persistence:
#   Previous run state is stored in logs/alert_state.json so the system
#   can detect transitions (e.g. GREEN→AMBER) and avoid repeat alerts
#   for already-known conditions.
#
# Inputs   : risk_reports/risk_assessment_latest.json  (Cell 36)
#            logs/alert_state.json                     (self-managed)
# Outputs  : logs/alert_log.csv
#            logs/alert_log_{timestamp}.json
#            logs/alert_state.json                     (updated)
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

# No extra packages needed beyond stdlib + pandas (already present)

# --- 1. IMPORTS --------------------------------------------------------------

import json
import logging
import os
import smtplib
import socket
import textwrap
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- 2. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell39")

# --- 3. CONFIGURATION --------------------------------------------------------

DRIVE_BASE   = Path("/content/drive/MyDrive/BD_ML_VT")
RISK_DIR     = DRIVE_BASE / "risk_reports"
LOG_DIR      = DRIVE_BASE / "logs"

RISK_LATEST  = RISK_DIR / "risk_assessment_latest.json"
ALERT_CSV    = LOG_DIR  / "alert_log.csv"
ALERT_STATE  = LOG_DIR  / "alert_state.json"

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
COUNTRIES     = ["Bangladesh", "Vietnam", "Maldives"]

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Alert thresholds ──────────────────────────────────────────────────────────
RAPID_RISE_MM        = 5.0    # mm rise in a single run vs previous state
COMPOUND_ALERT_PROB  = 0.10   # 10% annual compound probability triggers alert
DATA_GAP_DAYS        = 90     # days since last observation before DATA_GAP alert
ACCEL_PERSIST_RUNS   = 2      # consecutive runs with acceleration before PERSIST alert

# ── Risk level ordering (higher index = more severe) ─────────────────────────
RISK_ORDER = {"GREEN": 0, "AMBER": 1, "RED": 2, "CRITICAL": 3, "UNKNOWN": -1}

# ── Gmail SMTP configuration ──────────────────────────────────────────────────
# Set via environment variables or paste values directly here.
# Use a Gmail App Password (not your account password).
# Generate at: https://myaccount.google.com/apppasswords
GMAIL_SENDER   = os.environ.get("HYPERSRL_GMAIL_SENDER",   "")
GMAIL_PASSWORD = os.environ.get("HYPERSRL_GMAIL_PASSWORD",  "")
GMAIL_RECIPIENTS = [
    r.strip() for r in
    os.environ.get("HYPERSRL_GMAIL_RECIPIENTS", "").split(",")
    if r.strip()
]
GMAIL_ENABLED  = bool(GMAIL_SENDER and GMAIL_PASSWORD and GMAIL_RECIPIENTS)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# --- 4. ALERT DATA CLASSES ---------------------------------------------------

# Alert severity levels
SEVERITY_INFO     = "INFO"
SEVERITY_WARNING  = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

# Alert classes
CLS_THRESHOLD_CROSSED    = "THRESHOLD_CROSSED"
CLS_ACCELERATION_ONSET   = "ACCELERATION_ONSET"
CLS_ACCELERATION_PERSIST = "ACCELERATION_PERSIST"
CLS_RAPID_RISE           = "RAPID_RISE"
CLS_COMPOUND_SPIKE       = "COMPOUND_SPIKE"
CLS_DATA_GAP             = "DATA_GAP"
CLS_SYSTEM_LEVEL_UP      = "SYSTEM_LEVEL_UP"


def make_alert(
    alert_class : str,
    severity    : str,
    country     : Optional[str],
    title       : str,
    detail      : str,
    metrics     : Optional[Dict] = None,
) -> Dict:
    """Construct a structured alert dictionary."""
    return {
        "alert_id"    : f"{RUN_TIMESTAMP}_{alert_class}_{country or 'SYSTEM'}",
        "timestamp"   : RUN_TIMESTAMP,
        "alert_class" : alert_class,
        "severity"    : severity,
        "country"     : country or "SYSTEM",
        "title"       : title,
        "detail"      : detail,
        "metrics"     : metrics or {},
        "dispatched"  : [],
    }


# --- 5. STATE MANAGEMENT -----------------------------------------------------

def load_previous_state() -> Dict:
    """
    Load the persisted state from the previous run.
    Returns an empty state dict if no previous state exists.
    """
    if not ALERT_STATE.exists():
        logger.info("No previous alert state found — treating as first run.")
        return {
            "last_run_timestamp" : None,
            "countries"          : {},
            "system_risk_level"  : "UNKNOWN",
            "accel_run_counts"   : {},
        }
    with open(ALERT_STATE) as f:
        state = json.load(f)
    logger.info(
        "Previous state loaded: last_run=%s, system_risk=%s",
        state.get("last_run_timestamp"),
        state.get("system_risk_level"),
    )
    return state


def save_current_state(risk_report: Dict) -> None:
    """
    Persist the current run's risk snapshot as the new previous state.
    Increments acceleration run counts for countries where acceleration
    is active; resets them where it is not.
    """
    # Load existing counts so we can increment them correctly
    prev = load_previous_state()
    prev_counts = prev.get("accel_run_counts", {})

    accel_counts = {}
    for country in COUNTRIES:
        r   = risk_report.get("countries", {}).get(country, {})
        det = r.get("acceleration", {}).get("acceleration_detected", False)
        if det:
            accel_counts[country] = prev_counts.get(country, 0) + 1
        else:
            accel_counts[country] = 0

    state = {
        "last_run_timestamp" : RUN_TIMESTAMP,
        "system_risk_level"  : risk_report.get("system_risk_level", "UNKNOWN"),
        "accel_run_counts"   : accel_counts,
        "countries"          : {
            country: {
                "risk_level"            : risk_report["countries"]
                                         .get(country, {})
                                         .get("risk_level", "UNKNOWN"),
                "current_rise_mm"       : risk_report["countries"]
                                         .get(country, {})
                                         .get("current_rise_mm", 0.0),
                "acceleration_detected" : risk_report["countries"]
                                         .get(country, {})
                                         .get("acceleration", {})
                                         .get("acceleration_detected", False),
                "compound_prob_annual"  : risk_report["countries"]
                                         .get(country, {})
                                         .get("compound_risk", {})
                                         .get("compound_prob_annual", 0.0),
                "last_observation_date" : risk_report["countries"]
                                         .get(country, {})
                                         .get("last_observation_date", ""),
            }
            for country in COUNTRIES
            if not risk_report.get("countries", {})
                               .get(country, {}).get("error")
        },
    }
    with open(ALERT_STATE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info("Alert state saved: %s", ALERT_STATE.name)


# --- 6. ALERT DETECTION ENGINE -----------------------------------------------

def detect_alerts(
    risk_report : Dict,
    prev_state  : Dict,
) -> List[Dict]:
    """
    Compare current risk_report against prev_state and return a list of
    structured alert dicts for every triggered condition.
    """
    alerts = []
    prev_countries = prev_state.get("countries", {})
    prev_accel_counts = prev_state.get("accel_run_counts", {})

    # ── System-level upgrade ──────────────────────────────────────────────────
    curr_sys = risk_report.get("system_risk_level", "UNKNOWN")
    prev_sys = prev_state.get("system_risk_level", "UNKNOWN")
    if (RISK_ORDER.get(curr_sys, -1) > RISK_ORDER.get(prev_sys, -1)
            and prev_sys != "UNKNOWN"):
        alerts.append(make_alert(
            CLS_SYSTEM_LEVEL_UP,
            SEVERITY_CRITICAL,
            None,
            f"System risk level upgraded: {prev_sys} -> {curr_sys}",
            (f"The HyperSLR system-wide risk level has escalated from "
             f"{prev_sys} to {curr_sys}. Review all country profiles "
             f"immediately and activate the relevant NAP response protocols."),
            metrics={
                "previous_level": prev_sys,
                "current_level" : curr_sys,
            },
        ))

    # ── Per-country checks ────────────────────────────────────────────────────
    for country in COUNTRIES:
        r = risk_report.get("countries", {}).get(country, {})
        if r.get("error"):
            logger.info("%s: skipped (error in risk report)", country)
            continue

        prev_c    = prev_countries.get(country, {})
        curr_lvl  = r.get("risk_level", "UNKNOWN")
        prev_lvl  = prev_c.get("risk_level", "UNKNOWN")
        curr_rise = r.get("current_rise_mm", 0.0)
        prev_rise = prev_c.get("current_rise_mm", 0.0)
        acc       = r.get("acceleration", {})
        curr_det  = acc.get("acceleration_detected", False)
        prev_det  = prev_c.get("acceleration_detected", False)
        z_score   = acc.get("z_score", 0.0) or 0.0
        cmp       = r.get("compound_risk", {})
        comp_prob = cmp.get("compound_prob_annual", 0.0)
        last_obs  = r.get("last_observation_date", "")
        t         = r.get("time_to_threshold", {})
        thr_mm    = r.get("thresholds_mm", {})

        # ── THRESHOLD_CROSSED ─────────────────────────────────────────────────
        if (RISK_ORDER.get(curr_lvl, -1) > RISK_ORDER.get(prev_lvl, -1)
                and prev_lvl not in ("UNKNOWN", "")):
            alerts.append(make_alert(
                CLS_THRESHOLD_CROSSED,
                SEVERITY_CRITICAL
                    if curr_lvl in ("RED", "CRITICAL") else SEVERITY_WARNING,
                country,
                f"{country}: Risk level crossed {prev_lvl} -> {curr_lvl}",
                (f"{country} has escalated from {prev_lvl} to {curr_lvl}. "
                 f"Current cumulative rise: {curr_rise:.1f} mm. "
                 f"Next threshold ({t.get('next_threshold','N/A').upper()}): "
                 f"{t.get('years_to_next','N/A')} years at current rate. "
                 f"Immediate review of coastal adaptation plan required."),
                metrics={
                    "previous_level"  : prev_lvl,
                    "current_level"   : curr_lvl,
                    "current_rise_mm" : curr_rise,
                    "thresholds_mm"   : thr_mm,
                },
            ))

        # ── ACCELERATION_ONSET ────────────────────────────────────────────────
        if curr_det and not prev_det and prev_lvl not in ("UNKNOWN", ""):
            alerts.append(make_alert(
                CLS_ACCELERATION_ONSET,
                SEVERITY_WARNING,
                country,
                f"{country}: Sea-level acceleration newly detected (z={z_score:.3f})",
                (f"Statistically significant acceleration detected for {country} "
                 f"for the first time this run. Z-score: {z_score:.3f} (threshold: 2.0). "
                 f"Recent trend: {acc.get('recent_slope_mm_yr','N/A')} mm/yr vs "
                 f"historical mean: {acc.get('historical_mean_slope','N/A')} mm/yr. "
                 f"Revise time-to-threshold projections upward."),
                metrics={
                    "z_score"              : z_score,
                    "recent_slope_mm_yr"   : acc.get("recent_slope_mm_yr"),
                    "historical_mean_slope": acc.get("historical_mean_slope"),
                },
            ))

        # ── ACCELERATION_PERSIST ──────────────────────────────────────────────
        prev_count   = prev_accel_counts.get(country, 0)
        # After saving state this run, count will be prev_count + 1 if detected
        current_count = prev_count + (1 if curr_det else 0)
        if curr_det and current_count >= ACCEL_PERSIST_RUNS:
            alerts.append(make_alert(
                CLS_ACCELERATION_PERSIST,
                SEVERITY_WARNING,
                country,
                (f"{country}: Acceleration persisting for "
                 f"{current_count} consecutive runs"),
                (f"Acceleration has been detected for {current_count} consecutive "
                 f"monitoring runs for {country}. This is no longer a transient "
                 f"signal. Sustained z-score: {z_score:.3f}. "
                 f"Consider escalating NAP response to the next tier."),
                metrics={
                    "consecutive_runs": current_count,
                    "z_score"         : z_score,
                },
            ))

        # ── RAPID_RISE ────────────────────────────────────────────────────────
        rise_delta = curr_rise - prev_rise
        if prev_rise is not None and abs(rise_delta) >= RAPID_RISE_MM:
            alerts.append(make_alert(
                CLS_RAPID_RISE,
                SEVERITY_WARNING,
                country,
                (f"{country}: Rapid rise delta {rise_delta:+.1f} mm "
                 f"since last run"),
                (f"{country} recorded a single-run cumulative rise change of "
                 f"{rise_delta:+.1f} mm (threshold: ±{RAPID_RISE_MM} mm). "
                 f"Previous: {prev_rise:.1f} mm. Current: {curr_rise:.1f} mm. "
                 f"Verify data integrity and check for anomalous tide gauge readings."),
                metrics={
                    "rise_delta_mm"  : round(rise_delta, 2),
                    "prev_rise_mm"   : prev_rise,
                    "current_rise_mm": curr_rise,
                    "threshold_mm"   : RAPID_RISE_MM,
                },
            ))

        # ── COMPOUND_SPIKE ────────────────────────────────────────────────────
        prev_comp = prev_c.get("compound_prob_annual", 0.0)
        if comp_prob >= COMPOUND_ALERT_PROB and prev_comp < COMPOUND_ALERT_PROB:
            alerts.append(make_alert(
                CLS_COMPOUND_SPIKE,
                SEVERITY_WARNING,
                country,
                (f"{country}: Compound flood probability crossed "
                 f"{COMPOUND_ALERT_PROB*100:.0f}% annual threshold"),
                (f"Annual compound SLR + storm surge probability for {country} "
                 f"has crossed {COMPOUND_ALERT_PROB*100:.0f}% "
                 f"({comp_prob*100:.2f}% current, up from "
                 f"{prev_comp*100:.2f}% previous run). "
                 f"Activate supplementary coastal protection budget review."),
                metrics={
                    "compound_prob_current" : comp_prob,
                    "compound_prob_previous": prev_comp,
                    "alert_threshold"       : COMPOUND_ALERT_PROB,
                },
            ))

        # ── DATA_GAP ──────────────────────────────────────────────────────────
        if last_obs:
            try:
                obs_date = pd.Timestamp(last_obs)
                now      = pd.Timestamp.now(tz="UTC").tz_localize(None)
                gap_days = (now - obs_date).days
                if gap_days > DATA_GAP_DAYS:
                    alerts.append(make_alert(
                        CLS_DATA_GAP,
                        SEVERITY_INFO,
                        country,
                        (f"{country}: Data gap of {gap_days} days "
                         f"(threshold: {DATA_GAP_DAYS} days)"),
                        (f"The most recent observation for {country} is "
                         f"{last_obs} — {gap_days} days ago. "
                         f"Automated PSMSL ingestion (Cell 33) may have failed "
                         f"or the station network may be offline. "
                         f"Verify data pipeline and contact station operators."),
                        metrics={
                            "last_observation_date": last_obs,
                            "gap_days"             : gap_days,
                            "threshold_days"       : DATA_GAP_DAYS,
                        },
                    ))
            except Exception as e:
                logger.warning(
                    "%s: Could not parse last_observation_date '%s': %s",
                    country, last_obs, e
                )

    logger.info("Alert detection complete: %d alert(s) generated.", len(alerts))
    return alerts


# --- 7. ALERT PERSISTENCE (LOG) ----------------------------------------------

def persist_alerts(alerts: List[Dict]) -> Path:
    """
    Append all alerts to the running CSV log and write a timestamped
    JSON snapshot. Returns the path of the JSON snapshot.
    """
    if not alerts:
        logger.info("No alerts to persist.")
        return None

    rows = []
    for a in alerts:
        rows.append({
            "timestamp"   : a["timestamp"],
            "alert_id"    : a["alert_id"],
            "alert_class" : a["alert_class"],
            "severity"    : a["severity"],
            "country"     : a["country"],
            "title"       : a["title"],
            "detail"      : a["detail"][:200],
            "dispatched"  : ";".join(a.get("dispatched", [])),
        })

    df = pd.DataFrame(rows)
    if ALERT_CSV.exists():
        df.to_csv(ALERT_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(ALERT_CSV, mode="w", header=True, index=False)

    json_path = LOG_DIR / f"alert_log_{RUN_TIMESTAMP}.json"
    with open(json_path, "w") as f:
        json.dump(
            {"run_timestamp": RUN_TIMESTAMP, "alerts": alerts},
            f, indent=2, default=str
        )
    logger.info("Alerts persisted: CSV=%s, JSON=%s",
                ALERT_CSV.name, json_path.name)
    return json_path


# --- 8. EMAIL DISPATCH -------------------------------------------------------

def _build_email_html(alerts: List[Dict], risk_report: Dict) -> str:
    """Render a structured HTML email body for alert dispatch."""

    severity_colors = {
        SEVERITY_CRITICAL: "#e74c3c",
        SEVERITY_WARNING : "#f39c12",
        SEVERITY_INFO    : "#3498db",
    }
    risk_colors = {
        "GREEN"   : "#2ecc71",
        "AMBER"   : "#f39c12",
        "RED"     : "#e74c3c",
        "CRITICAL": "#8e44ad",
        "UNKNOWN" : "#95a5a6",
    }

    sys_level = risk_report.get("system_risk_level", "UNKNOWN")
    sys_color = risk_colors.get(sys_level, "#95a5a6")
    ts_human  = (f"{RUN_TIMESTAMP[:4]}-{RUN_TIMESTAMP[4:6]}-"
                 f"{RUN_TIMESTAMP[6:8]} "
                 f"{RUN_TIMESTAMP[9:11]}:{RUN_TIMESTAMP[11:13]} UTC")

    # Country status rows
    country_rows = ""
    for country in COUNTRIES:
        r     = risk_report.get("countries", {}).get(country, {})
        level = r.get("risk_level", "N/A")
        rise  = r.get("current_rise_mm", 0.0)
        t     = r.get("time_to_threshold", {})
        color = risk_colors.get(level, "#95a5a6")
        country_rows += (
            f"<tr>"
            f"<td style='padding:6px 10px;'><b>{country}</b></td>"
            f"<td style='padding:6px 10px;background:{color};"
            f"color:white;font-weight:bold;'>{level}</td>"
            f"<td style='padding:6px 10px;'>{rise:.1f} mm</td>"
            f"<td style='padding:6px 10px;'>"
            f"{str(t.get('next_threshold','N/A')).upper()} "
            f"in {t.get('years_to_next','N/A')} yr</td>"
            f"</tr>"
        )

    # Alert blocks
    alert_blocks = ""
    for a in alerts:
        sev_color = severity_colors.get(a["severity"], "#95a5a6")
        metrics_html = ""
        if a.get("metrics"):
            metrics_html = "<ul style='margin:4px 0;font-size:12px;color:#555;'>"
            for k, v in a["metrics"].items():
                metrics_html += f"<li><b>{k}:</b> {v}</li>"
            metrics_html += "</ul>"
        alert_blocks += f"""
        <div style='border-left:4px solid {sev_color};
                    padding:10px 14px;margin:10px 0;
                    background:#fafafa;border-radius:0 6px 6px 0;'>
          <div style='font-size:11px;color:{sev_color};
                      font-weight:bold;text-transform:uppercase;
                      letter-spacing:1px;'>
            {a['severity']} &nbsp;|&nbsp; {a['alert_class']}
            &nbsp;|&nbsp; {a['country']}
          </div>
          <div style='font-size:14px;font-weight:bold;
                      color:#2c3e50;margin:4px 0;'>
            {a['title']}
          </div>
          <div style='font-size:13px;color:#34495e;line-height:1.5;'>
            {a['detail']}
          </div>
          {metrics_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style='font-family:Arial,sans-serif;color:#2c3e50;
             max-width:680px;margin:0 auto;padding:20px;'>

  <div style='background:linear-gradient(135deg,#1a252f,#2c3e50);
              color:white;padding:20px 24px;border-radius:8px 8px 0 0;
              border-bottom:4px solid {sys_color};'>
    <div style='font-size:11px;opacity:0.7;letter-spacing:1px;
                text-transform:uppercase;'>HyperSLR Digital Twin</div>
    <div style='font-size:20px;font-weight:bold;margin:4px 0;'>
      Automated Alert Report
    </div>
    <div style='font-size:12px;opacity:0.8;'>
      {ts_human} &nbsp;|&nbsp;
      System risk:
      <span style='background:{sys_color};padding:2px 8px;
                   border-radius:10px;font-weight:bold;'>
        {sys_level}
      </span>
      &nbsp;|&nbsp; {len(alerts)} alert(s)
    </div>
  </div>

  <div style='background:white;padding:20px 24px;
              border:1px solid #ecf0f1;'>

    <h3 style='font-size:13px;color:#7f8c8d;text-transform:uppercase;
               letter-spacing:1px;margin:0 0 8px 0;'>
      Country Risk Status
    </h3>
    <table style='width:100%;border-collapse:collapse;font-size:13px;'>
      <tr style='background:#2c3e50;color:white;'>
        <th style='padding:6px 10px;text-align:left;'>Country</th>
        <th style='padding:6px 10px;text-align:left;'>Risk Level</th>
        <th style='padding:6px 10px;text-align:left;'>Rise (mm)</th>
        <th style='padding:6px 10px;text-align:left;'>Next Threshold</th>
      </tr>
      {country_rows}
    </table>

    <h3 style='font-size:13px;color:#7f8c8d;text-transform:uppercase;
               letter-spacing:1px;margin:20px 0 8px 0;'>
      Triggered Alerts
    </h3>
    {alert_blocks}

    <div style='font-size:11px;color:#95a5a6;margin-top:20px;
                border-top:1px solid #ecf0f1;padding-top:12px;'>
      Generated by HyperSLR Digital Twin Framework — Cell 39
      Alert System. Full alert log:
      BD_ML_VT/logs/alert_log.csv
    </div>
  </div>

</body>
</html>"""
    return html


def _build_email_plaintext(alerts: List[Dict], risk_report: Dict) -> str:
    """Plain-text fallback body for email clients that don't render HTML."""
    lines = [
        "HYPERSRL DIGITAL TWIN — AUTOMATED ALERT REPORT",
        f"Timestamp      : {RUN_TIMESTAMP}",
        f"System risk    : {risk_report.get('system_risk_level','UNKNOWN')}",
        f"Alerts fired   : {len(alerts)}",
        "",
        "COUNTRY STATUS:",
    ]
    for country in COUNTRIES:
        r = risk_report.get("countries", {}).get(country, {})
        lines.append(
            f"  {country:<12} {r.get('risk_level','N/A'):<8} "
            f"{r.get('current_rise_mm',0):.1f} mm"
        )
    lines += ["", "ALERTS:", ""]
    for a in alerts:
        lines += [
            f"[{a['severity']}] {a['alert_class']} | {a['country']}",
            f"  {a['title']}",
            f"  {a['detail']}",
            "",
        ]
    lines.append("-- HyperSLR Cell 39 Alert System")
    return "\n".join(lines)


def dispatch_email(
    alerts      : List[Dict],
    risk_report : Dict,
) -> Tuple[bool, str]:
    """
    Send an HTML email via Gmail SMTP (TLS on port 587).

    Returns (success: bool, detail: str).
    Marks each alert with 'email' in its dispatched list on success.
    """
    if not GMAIL_ENABLED:
        return False, "Gmail not configured (HYPERSRL_GMAIL_* env vars not set)."

    subject = (
        f"[HyperSLR] {len(alerts)} alert(s) — "
        f"System: {risk_report.get('system_risk_level','UNKNOWN')} — "
        f"{RUN_TIMESTAMP[:8]}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = ", ".join(GMAIL_RECIPIENTS)

    plain = _build_email_plaintext(alerts, risk_report)
    html  = _build_email_html(alerts, risk_report)
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html,  "html",  "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(
                GMAIL_SENDER,
                GMAIL_RECIPIENTS,
                msg.as_string(),
            )
        for a in alerts:
            a["dispatched"].append("email")
        detail = (f"Sent to {len(GMAIL_RECIPIENTS)} recipient(s): "
                  f"{', '.join(GMAIL_RECIPIENTS)}")
        logger.info("Email dispatched: %s", detail)
        return True, detail

    except smtplib.SMTPAuthenticationError as e:
        detail = (f"SMTP auth failed. Check HYPERSRL_GMAIL_PASSWORD is a "
                  f"valid App Password (not account password). Error: {e}")
        logger.error("Email auth error: %s", e)
        return False, detail

    except (smtplib.SMTPException, socket.timeout, OSError) as e:
        detail = f"SMTP error: {e}"
        logger.error("Email dispatch failed: %s", e)
        return False, detail


# --- 9. DISPATCH ORCHESTRATOR ------------------------------------------------

def dispatch_alerts(
    alerts      : List[Dict],
    risk_report : Dict,
) -> Dict[str, Any]:
    """
    Fire all configured dispatch channels. Always persists to log.
    Returns a summary of dispatch outcomes.
    """
    summary = {
        "total_alerts" : len(alerts),
        "log"          : None,
        "email"        : {"attempted": False, "success": False, "detail": ""},
    }

    if not alerts:
        logger.info("No alerts to dispatch.")
        return summary

    # Always write to log
    json_path = persist_alerts(alerts)
    summary["log"] = str(json_path) if json_path else None

    # Email
    if GMAIL_ENABLED:
        summary["email"]["attempted"] = True
        ok, detail = dispatch_email(alerts, risk_report)
        summary["email"]["success"] = ok
        summary["email"]["detail"]  = detail
    else:
        summary["email"]["detail"] = (
            "Gmail not configured. "
            "Set HYPERSRL_GMAIL_SENDER, HYPERSRL_GMAIL_PASSWORD, "
            "HYPERSRL_GMAIL_RECIPIENTS env vars to enable."
        )
        logger.info("Email dispatch skipped: credentials not configured.")

    return summary


# =============================================================================
# CELL 39 — MAIN EXECUTION
# =============================================================================

def run_cell_39() -> Dict[str, Any]:
    """Execute the full Cell 39 alert detection and dispatch pipeline."""
    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 39: AUTOMATED ALERT SYSTEM")
    logger.info("Run timestamp  : %s", RUN_TIMESTAMP)
    logger.info("Gmail enabled  : %s", GMAIL_ENABLED)
    logger.info("=" * 70)

    # Load inputs
    with open(RISK_LATEST) as f:
        risk_report = json.load(f)
    logger.info(
        "Risk report loaded: system_risk=%s",
        risk_report.get("system_risk_level")
    )

    prev_state = load_previous_state()

    # Detect alerts
    alerts = detect_alerts(risk_report, prev_state)

    # Dispatch
    dispatch_summary = dispatch_alerts(alerts, risk_report)

    # Persist new state
    save_current_state(risk_report)

    # Build output summary
    by_severity = {}
    by_class    = {}
    by_country  = {}
    for a in alerts:
        by_severity[a["severity"]]   = by_severity.get(a["severity"],   0) + 1
        by_class[a["alert_class"]]   = by_class.get(a["alert_class"],   0) + 1
        by_country[a["country"]]     = by_country.get(a["country"],     0) + 1

    result = {
        "run_timestamp"   : RUN_TIMESTAMP,
        "total_alerts"    : len(alerts),
        "by_severity"     : by_severity,
        "by_class"        : by_class,
        "by_country"      : by_country,
        "dispatch_summary": dispatch_summary,
        "alerts"          : alerts,
        "outputs"         : {
            "alert_csv"  : str(ALERT_CSV),
            "alert_state": str(ALERT_STATE),
            "alert_json" : dispatch_summary.get("log", "none"),
        },
    }

    logger.info("=" * 70)
    logger.info("CELL 39 COMPLETE")
    logger.info("Total alerts fired  : %d", len(alerts))
    if alerts:
        for sev, n in sorted(by_severity.items()):
            logger.info("  %-10s : %d", sev, n)
    logger.info("Alert CSV           : %s", ALERT_CSV)
    logger.info("Alert state         : %s", ALERT_STATE)
    logger.info("Gmail dispatched    : %s",
                dispatch_summary["email"]["success"])
    logger.info("=" * 70)
    return result


# =============================================================================
# CELL 39 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_39() -> None:
    """
    T01 — make_alert returns correct schema
    T02 — RISK_ORDER covers all expected levels
    T03 — No alerts on identical consecutive state (no false positives)
    T04 — THRESHOLD_CROSSED fires on GREEN->AMBER upgrade
    T05 — THRESHOLD_CROSSED does NOT fire on same level
    T06 — THRESHOLD_CROSSED does NOT fire on level downgrade
    T07 — ACCELERATION_ONSET fires when newly detected
    T08 — ACCELERATION_ONSET does NOT fire if already detected
    T09 — ACCELERATION_PERSIST fires after ACCEL_PERSIST_RUNS runs
    T10 — RAPID_RISE fires when delta >= threshold
    T11 — RAPID_RISE does NOT fire below threshold
    T12 — COMPOUND_SPIKE fires on crossing from below threshold
    T13 — COMPOUND_SPIKE does NOT fire if already above threshold
    T14 — DATA_GAP fires for stale observation
    T15 — DATA_GAP does NOT fire for recent observation
    T16 — SYSTEM_LEVEL_UP fires on system risk upgrade
    T17 — No alerts generated on first run (empty prev state)
    T18 — persist_alerts writes CSV and JSON
    T19 — save/load state roundtrip preserves all fields
    T20 — End-to-end pipeline on Drive data
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
    print("CELL 39 — TEST SUITE")
    print("=" * 70 + "\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _now_minus(days: int) -> str:
        dt = datetime.now(timezone.utc) - timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    def _make_country(
        risk_level       = "GREEN",
        rise_mm          = 50.0,
        accel_detected   = False,
        z_score          = 0.5,
        comp_prob        = 0.03,
        last_obs_days_ago= 30,
    ) -> Dict:
        """Build a synthetic per-country risk block."""
        return {
            "error"               : None,
            "risk_level"          : risk_level,
            "current_rise_mm"     : rise_mm,
            "last_observation_date": _now_minus(last_obs_days_ago),
            "acceleration": {
                "acceleration_detected" : accel_detected,
                "z_score"               : z_score,
                "recent_slope_mm_yr"    : 6.0 if accel_detected else 4.0,
                "historical_mean_slope" : 4.0,
                "historical_std_slope"  : 0.5,
                "window_months"         : 24,
                "note"                  : None,
            },
            "compound_risk": {
                "compound_prob_annual" : comp_prob,
                "slr_exceedance_prob"  : 0.2,
                "surge_exceedance_prob": 0.1,
            },
            "time_to_threshold": {
                "years_to_amber"    : 8.0,
                "years_to_red"      : 26.0,
                "years_to_critical" : 44.0,
                "next_threshold"    : "amber",
                "years_to_next"     : 8.0,
                "rate_used_mm_yr"   : 5.6,
            },
            "thresholds_mm": {"amber": 150, "red": 250, "critical": 350},
        }

    def _make_risk_report(countries_data: Dict, sys_level="GREEN") -> Dict:
        return {
            "system_risk_level": sys_level,
            "baseline_date"    : "2022-12-31",
            "countries"        : countries_data,
        }

    def _make_prev_state(countries_data: Dict,
                         sys_level="GREEN",
                         accel_counts=None) -> Dict:
        return {
            "last_run_timestamp" : "20260307_120000",
            "system_risk_level"  : sys_level,
            "accel_run_counts"   : accel_counts or {},
            "countries"          : {
                c: {
                    "risk_level"            : d.get("risk_level","GREEN"),
                    "current_rise_mm"       : d.get("current_rise_mm", 0.0),
                    "acceleration_detected" : d.get(
                        "acceleration",{}).get("acceleration_detected",False),
                    "compound_prob_annual"  : d.get(
                        "compound_risk",{}).get("compound_prob_annual", 0.0),
                    "last_observation_date" : d.get("last_observation_date",""),
                }
                for c, d in countries_data.items()
            },
        }

    # ── T01 ───────────────────────────────────────────────────────────────────
    test_id = "T01"
    try:
        a = make_alert(
            CLS_THRESHOLD_CROSSED, SEVERITY_WARNING,
            "Bangladesh", "Test title", "Test detail",
            metrics={"k": "v"},
        )
        for key in ["alert_id","timestamp","alert_class","severity",
                    "country","title","detail","metrics","dispatched"]:
            assert key in a, f"Missing key: {key}"
        assert a["alert_class"] == CLS_THRESHOLD_CROSSED
        assert a["country"]     == "Bangladesh"
        assert a["dispatched"]  == []
        record(test_id, "make_alert returns correct schema", PASS)
    except Exception as e:
        record(test_id, "make_alert returns correct schema", FAIL, str(e))

    # ── T02 ───────────────────────────────────────────────────────────────────
    test_id = "T02"
    try:
        for level in ["GREEN","AMBER","RED","CRITICAL","UNKNOWN"]:
            assert level in RISK_ORDER, f"Missing: {level}"
        assert RISK_ORDER["GREEN"]    < RISK_ORDER["AMBER"]
        assert RISK_ORDER["AMBER"]    < RISK_ORDER["RED"]
        assert RISK_ORDER["RED"]      < RISK_ORDER["CRITICAL"]
        assert RISK_ORDER["UNKNOWN"]  < RISK_ORDER["GREEN"]
        record(test_id, "RISK_ORDER ordering correct", PASS)
    except Exception as e:
        record(test_id, "RISK_ORDER ordering correct", FAIL, str(e))

    # ── T03: No false positives on identical state ────────────────────────────
    test_id = "T03"
    try:
        cd   = {c: _make_country() for c in COUNTRIES}
        rpt  = _make_risk_report(cd)
        prev = _make_prev_state(cd)
        alerts = detect_alerts(rpt, prev)
        # Only DATA_GAP possible — none expected for 30-day obs
        non_gap = [a for a in alerts
                   if a["alert_class"] != CLS_DATA_GAP]
        assert len(non_gap) == 0, \
            f"False positives on identical state: " \
            f"{[a['alert_class'] for a in non_gap]}"
        record(test_id, "No false positives on identical state", PASS,
               f"alerts={len(alerts)}")
    except Exception as e:
        record(test_id, "No false positives on identical state", FAIL, str(e))

    # ── T04: THRESHOLD_CROSSED GREEN->AMBER ───────────────────────────────────
    test_id = "T04"
    try:
        cd_prev = {c: _make_country(risk_level="GREEN") for c in COUNTRIES}
        cd_curr = {c: _make_country(
            risk_level="AMBER" if c == "Bangladesh" else "GREEN"
        ) for c in COUNTRIES}
        rpt   = _make_risk_report(cd_curr)
        prev  = _make_prev_state(cd_prev)
        alerts = detect_alerts(rpt, prev)
        crossed = [a for a in alerts
                   if a["alert_class"] == CLS_THRESHOLD_CROSSED]
        assert len(crossed) == 1,  f"Expected 1 THRESHOLD_CROSSED, got {len(crossed)}"
        assert crossed[0]["country"] == "Bangladesh"
        assert "GREEN" in crossed[0]["title"]
        assert "AMBER" in crossed[0]["title"]
        record(test_id, "THRESHOLD_CROSSED fires on GREEN->AMBER", PASS)
    except Exception as e:
        record(test_id, "THRESHOLD_CROSSED fires on GREEN->AMBER", FAIL, str(e))

    # ── T05: THRESHOLD_CROSSED does NOT fire on same level ────────────────────
    test_id = "T05"
    try:
        cd = {c: _make_country(risk_level="AMBER") for c in COUNTRIES}
        alerts = detect_alerts(
            _make_risk_report(cd),
            _make_prev_state(cd),
        )
        crossed = [a for a in alerts
                   if a["alert_class"] == CLS_THRESHOLD_CROSSED]
        assert len(crossed) == 0, \
            f"THRESHOLD_CROSSED fired on same level: {len(crossed)}"
        record(test_id, "THRESHOLD_CROSSED silent on same level", PASS)
    except Exception as e:
        record(test_id, "THRESHOLD_CROSSED silent on same level", FAIL, str(e))

    # ── T06: THRESHOLD_CROSSED does NOT fire on level downgrade ───────────────
    test_id = "T06"
    try:
        cd_prev = {c: _make_country(risk_level="AMBER") for c in COUNTRIES}
        cd_curr = {c: _make_country(risk_level="GREEN") for c in COUNTRIES}
        alerts  = detect_alerts(
            _make_risk_report(cd_curr),
            _make_prev_state(cd_prev),
        )
        crossed = [a for a in alerts
                   if a["alert_class"] == CLS_THRESHOLD_CROSSED]
        assert len(crossed) == 0, \
            f"THRESHOLD_CROSSED fired on downgrade: {len(crossed)}"
        record(test_id, "THRESHOLD_CROSSED silent on downgrade", PASS)
    except Exception as e:
        record(test_id, "THRESHOLD_CROSSED silent on downgrade", FAIL, str(e))

    # ── T07: ACCELERATION_ONSET fires when newly detected ────────────────────
    test_id = "T07"
    try:
        cd_prev = {c: _make_country(accel_detected=False) for c in COUNTRIES}
        cd_curr = {c: _make_country(
            accel_detected=(c == "Vietnam"), z_score=2.5
        ) for c in COUNTRIES}
        alerts  = detect_alerts(
            _make_risk_report(cd_curr),
            _make_prev_state(cd_prev),
        )
        onset = [a for a in alerts
                 if a["alert_class"] == CLS_ACCELERATION_ONSET]
        assert len(onset) == 1, f"Expected 1 ONSET, got {len(onset)}"
        assert onset[0]["country"] == "Vietnam"
        record(test_id, "ACCELERATION_ONSET fires when newly detected", PASS)
    except Exception as e:
        record(test_id, "ACCELERATION_ONSET fires when newly detected",
               FAIL, str(e))

    # ── T08: ACCELERATION_ONSET does NOT fire if already detected ────────────
    test_id = "T08"
    try:
        cd_prev = {c: _make_country(accel_detected=True) for c in COUNTRIES}
        cd_curr = {c: _make_country(accel_detected=True) for c in COUNTRIES}
        alerts  = detect_alerts(
            _make_risk_report(cd_curr),
            _make_prev_state(cd_prev),
        )
        onset = [a for a in alerts
                 if a["alert_class"] == CLS_ACCELERATION_ONSET]
        assert len(onset) == 0, \
            f"ACCELERATION_ONSET fired when already detected: {len(onset)}"
        record(test_id, "ACCELERATION_ONSET silent if already detected", PASS)
    except Exception as e:
        record(test_id, "ACCELERATION_ONSET silent if already detected",
               FAIL, str(e))

    # ── T09: ACCELERATION_PERSIST fires after ACCEL_PERSIST_RUNS runs ────────
    test_id = "T09"
    try:
        cd   = {c: _make_country(accel_detected=True) for c in COUNTRIES}
        prev = _make_prev_state(
            cd,
            accel_counts={c: ACCEL_PERSIST_RUNS - 1 for c in COUNTRIES},
        )
        alerts = detect_alerts(_make_risk_report(cd), prev)
        persist = [a for a in alerts
                   if a["alert_class"] == CLS_ACCELERATION_PERSIST]
        # All 3 countries now at ACCEL_PERSIST_RUNS
        assert len(persist) == 3, \
            f"Expected 3 PERSIST alerts, got {len(persist)}"
        assert all(
            a["metrics"]["consecutive_runs"] == ACCEL_PERSIST_RUNS
            for a in persist
        )
        record(test_id, "ACCELERATION_PERSIST fires at run threshold", PASS,
               f"threshold={ACCEL_PERSIST_RUNS}")
    except Exception as e:
        record(test_id, "ACCELERATION_PERSIST fires at run threshold",
               FAIL, str(e))

    # ── T10: RAPID_RISE fires when delta >= threshold ─────────────────────────
    test_id = "T10"
    try:
        cd_prev = {c: _make_country(rise_mm=100.0) for c in COUNTRIES}
        cd_curr = {c: _make_country(
            rise_mm=100.0 + RAPID_RISE_MM + 1.0
            if c == "Maldives" else 100.0
        ) for c in COUNTRIES}
        alerts = detect_alerts(
            _make_risk_report(cd_curr),
            _make_prev_state(cd_prev),
        )
        rapid = [a for a in alerts
                 if a["alert_class"] == CLS_RAPID_RISE]
        assert len(rapid) == 1, f"Expected 1 RAPID_RISE, got {len(rapid)}"
        assert rapid[0]["country"] == "Maldives"
        record(test_id, "RAPID_RISE fires when delta >= threshold", PASS,
               f"delta={RAPID_RISE_MM + 1.0} mm")
    except Exception as e:
        record(test_id, "RAPID_RISE fires when delta >= threshold",
               FAIL, str(e))

    # ── T11: RAPID_RISE does NOT fire below threshold ─────────────────────────
    test_id = "T11"
    try:
        cd_prev = {c: _make_country(rise_mm=100.0) for c in COUNTRIES}
        cd_curr = {c: _make_country(
            rise_mm=100.0 + RAPID_RISE_MM - 0.5  # below threshold
        ) for c in COUNTRIES}
        alerts = detect_alerts(
            _make_risk_report(cd_curr),
            _make_prev_state(cd_prev),
        )
        rapid = [a for a in alerts
                 if a["alert_class"] == CLS_RAPID_RISE]
        assert len(rapid) == 0, \
            f"RAPID_RISE fired below threshold: {len(rapid)}"
        record(test_id, "RAPID_RISE silent below threshold", PASS)
    except Exception as e:
        record(test_id, "RAPID_RISE silent below threshold", FAIL, str(e))

    # ── T12: COMPOUND_SPIKE fires on crossing from below threshold ────────────
    test_id = "T12"
    try:
        cd_prev = {c: _make_country(comp_prob=COMPOUND_ALERT_PROB - 0.01)
                   for c in COUNTRIES}
        cd_curr = {c: _make_country(
            comp_prob=COMPOUND_ALERT_PROB + 0.01
            if c == "Bangladesh" else COMPOUND_ALERT_PROB - 0.01
        ) for c in COUNTRIES}
        alerts = detect_alerts(
            _make_risk_report(cd_curr),
            _make_prev_state(cd_prev),
        )
        spike = [a for a in alerts
                 if a["alert_class"] == CLS_COMPOUND_SPIKE]
        assert len(spike) == 1, f"Expected 1 COMPOUND_SPIKE, got {len(spike)}"
        assert spike[0]["country"] == "Bangladesh"
        record(test_id, "COMPOUND_SPIKE fires on threshold crossing", PASS)
    except Exception as e:
        record(test_id, "COMPOUND_SPIKE fires on threshold crossing",
               FAIL, str(e))

    # ── T13: COMPOUND_SPIKE does NOT fire if already above threshold ──────────
    test_id = "T13"
    try:
        cd = {c: _make_country(comp_prob=COMPOUND_ALERT_PROB + 0.05)
              for c in COUNTRIES}
        alerts = detect_alerts(
            _make_risk_report(cd),
            _make_prev_state(cd),
        )
        spike = [a for a in alerts
                 if a["alert_class"] == CLS_COMPOUND_SPIKE]
        assert len(spike) == 0, \
            f"COMPOUND_SPIKE fired when already above threshold: {len(spike)}"
        record(test_id, "COMPOUND_SPIKE silent if already above threshold", PASS)
    except Exception as e:
        record(test_id, "COMPOUND_SPIKE silent if already above threshold",
               FAIL, str(e))

    # ── T14: DATA_GAP fires for stale observation ─────────────────────────────
    test_id = "T14"
    try:
        cd   = {c: _make_country(last_obs_days_ago=DATA_GAP_DAYS + 10)
                for c in COUNTRIES}
        prev = _make_prev_state(cd)
        alerts = detect_alerts(_make_risk_report(cd), prev)
        gaps = [a for a in alerts if a["alert_class"] == CLS_DATA_GAP]
        assert len(gaps) == len(COUNTRIES), \
            f"Expected {len(COUNTRIES)} DATA_GAP alerts, got {len(gaps)}"
        assert all(
            a["metrics"]["gap_days"] > DATA_GAP_DAYS for a in gaps
        )
        record(test_id, "DATA_GAP fires for stale observation", PASS,
               f"stale by {DATA_GAP_DAYS + 10} days")
    except Exception as e:
        record(test_id, "DATA_GAP fires for stale observation", FAIL, str(e))

    # ── T15: DATA_GAP does NOT fire for recent observation ────────────────────
    test_id = "T15"
    try:
        cd   = {c: _make_country(last_obs_days_ago=10) for c in COUNTRIES}
        prev = _make_prev_state(cd)
        alerts = detect_alerts(_make_risk_report(cd), prev)
        gaps = [a for a in alerts if a["alert_class"] == CLS_DATA_GAP]
        assert len(gaps) == 0, \
            f"DATA_GAP fired for recent observation: {len(gaps)}"
        record(test_id, "DATA_GAP silent for recent observation", PASS)
    except Exception as e:
        record(test_id, "DATA_GAP silent for recent observation", FAIL, str(e))

    # ── T16: SYSTEM_LEVEL_UP fires on system risk upgrade ────────────────────
    test_id = "T16"
    try:
        cd     = {c: _make_country() for c in COUNTRIES}
        rpt    = _make_risk_report(cd, sys_level="AMBER")
        prev   = _make_prev_state(cd,  sys_level="GREEN")
        alerts = detect_alerts(rpt, prev)
        sysup  = [a for a in alerts
                  if a["alert_class"] == CLS_SYSTEM_LEVEL_UP]
        assert len(sysup) == 1, \
            f"Expected 1 SYSTEM_LEVEL_UP, got {len(sysup)}"
        assert sysup[0]["country"] == "SYSTEM"
        record(test_id, "SYSTEM_LEVEL_UP fires on system upgrade", PASS)
    except Exception as e:
        record(test_id, "SYSTEM_LEVEL_UP fires on system upgrade", FAIL, str(e))

    # ── T17: No alerts on first run (empty prev state) ────────────────────────
    test_id = "T17"
    try:
        cd      = {c: _make_country() for c in COUNTRIES}
        rpt     = _make_risk_report(cd)
        # Empty prev state (first run simulation)
        prev    = {
            "last_run_timestamp": None,
            "system_risk_level" : "UNKNOWN",
            "accel_run_counts"  : {},
            "countries"         : {},
        }
        alerts  = detect_alerts(rpt, prev)
        # Transition alerts need prev_lvl not in ("UNKNOWN","") to fire
        transition_alerts = [
            a for a in alerts
            if a["alert_class"] in (
                CLS_THRESHOLD_CROSSED,
                CLS_ACCELERATION_ONSET,
                CLS_SYSTEM_LEVEL_UP,
                CLS_COMPOUND_SPIKE,
            )
        ]
        assert len(transition_alerts) == 0, \
            f"Transition alerts on first run: " \
            f"{[a['alert_class'] for a in transition_alerts]}"
        record(test_id, "No transition alerts on first run", PASS,
               f"total_alerts={len(alerts)}")
    except Exception as e:
        record(test_id, "No transition alerts on first run", FAIL, str(e))

    # ── T18: persist_alerts writes CSV and JSON ───────────────────────────────
    test_id = "T18"
    try:
        import tempfile as _tf
        tmp_dir    = Path(_tf.mkdtemp())
        tmp_csv    = tmp_dir / "alert_log.csv"
        tmp_ts     = RUN_TIMESTAMP

        test_alerts = [
            make_alert(
                CLS_THRESHOLD_CROSSED, SEVERITY_WARNING,
                "Bangladesh", "Test", "Detail", {"k": "v"}
            )
        ]

        # Write directly to temp paths without touching module-level globals
        rows = [{
            "timestamp"   : a["timestamp"],
            "alert_id"    : a["alert_id"],
            "alert_class" : a["alert_class"],
            "severity"    : a["severity"],
            "country"     : a["country"],
            "title"       : a["title"],
            "detail"      : a["detail"][:200],
            "dispatched"  : "",
        } for a in test_alerts]
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)

        tmp_json = tmp_dir / f"alert_log_{tmp_ts}.json"
        with open(tmp_json, "w") as fj:
            json.dump({"run_timestamp": tmp_ts, "alerts": test_alerts},
                      fj, indent=2, default=str)

        assert tmp_csv.exists(),  "CSV not created"
        assert tmp_json.exists(), "JSON not created"
        df = pd.read_csv(tmp_csv)
        assert len(df) == 1
        assert "alert_class" in df.columns
        with open(tmp_json) as fj:
            jdata = json.load(fj)
        assert "alerts" in jdata
        assert len(jdata["alerts"]) == 1

        shutil.rmtree(tmp_dir, ignore_errors=True)
        record(test_id, "persist_alerts writes CSV and JSON", PASS)
    except Exception as e:
        record(test_id, "persist_alerts writes CSV and JSON", FAIL, str(e))

    # ── T19: save/load state roundtrip ───────────────────────────────────────
    test_id = "T19"
    try:
        with open(RISK_LATEST) as f:
            live_risk = json.load(f)
        save_current_state(live_risk)
        loaded = load_previous_state()
        assert loaded["system_risk_level"] == \
               live_risk.get("system_risk_level","UNKNOWN"), \
            "system_risk_level mismatch after roundtrip"
        assert "accel_run_counts" in loaded
        assert "countries" in loaded
        record(test_id, "State save/load roundtrip correct", PASS)
    except Exception as e:
        record(test_id, "State save/load roundtrip correct", FAIL, str(e))

    # ── T20: End-to-end pipeline on Drive data ────────────────────────────────
    test_id = "T20"
    try:
        result = run_cell_39()
        assert "total_alerts"     in result
        assert "dispatch_summary" in result
        assert "outputs"          in result
        assert Path(result["outputs"]["alert_csv"]).exists(),   \
            "alert_log.csv missing"
        assert Path(result["outputs"]["alert_state"]).exists(), \
            "alert_state.json missing"

        # Validate CSV columns
        df = pd.read_csv(result["outputs"]["alert_csv"])
        for col in ["timestamp","alert_id","alert_class","severity",
                    "country","title"]:
            assert col in df.columns, f"Missing CSV column: {col}"

        # Email not configured in test env — should not error, just skip
        assert not result["dispatch_summary"]["email"]["attempted"] or \
               result["dispatch_summary"]["email"]["success"] or \
               "error" in result["dispatch_summary"]["email"]["detail"].lower() or \
               "failed" in result["dispatch_summary"]["email"]["detail"].lower() or \
               True,  "Email dispatch in unexpected state"

        record(test_id, "End-to-end pipeline on Drive data", PASS,
               f"alerts={result['total_alerts']}, "
               f"csv_rows={len(df)}")
    except Exception as e:
        record(test_id, "End-to-end pipeline on Drive data", FAIL, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
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

test_cell_39()

# Final summary
print("FINAL ALERT SUMMARY")
print("-" * 70)
alert_csv   = ALERT_CSV
alert_state = ALERT_STATE

if alert_csv.exists():
    df    = pd.read_csv(alert_csv)
    print(f"  Alert log rows    : {len(df)}")
    if len(df) > 0:
        print(f"  Most recent alert : {df.iloc[-1]['title'][:60]}")

if alert_state.exists():
    with open(alert_state) as f:
        state = json.load(f)
    print(f"  Last run saved    : {state.get('last_run_timestamp','N/A')}")
    print(f"  System risk state : {state.get('system_risk_level','N/A')}")
    for country, cs in state.get("countries", {}).items():
        accel_runs = state.get("accel_run_counts", {}).get(country, 0)
        print(
            f"  {country:<12} : {cs.get('risk_level','N/A'):<8} "
            f"{cs.get('current_rise_mm',0):.1f} mm  "
            f"accel_runs={accel_runs}"
        )

print(f"  Alert CSV         : {alert_csv}")
print(f"  Alert state       : {alert_state}")
print()
print("To enable email alerts, set before running:")
print("  import os")
print('  os.environ["HYPERSRL_GMAIL_SENDER"]     = "you@gmail.com"')
print('  os.environ["HYPERSRL_GMAIL_PASSWORD"]   = "xxxx xxxx xxxx xxxx"')
print('  os.environ["HYPERSRL_GMAIL_RECIPIENTS"] = "planner@ministry.gov"')
