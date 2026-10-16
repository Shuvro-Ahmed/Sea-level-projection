# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 38: Dashboard and Publication Report
# =============================================================================
# Purpose  : Render a static snapshot of all pipeline outputs as:
#              (A) Interactive HTML dashboard (Plotly, self-contained)
#              (B) Publication-quality PDF report (ReportLab + Matplotlib)
#            Both are generated from the same frozen data snapshot read
#            at execution time from Drive.
#
# Panels (7):
#   P1  SLR time series per country (historical + live, anomaly mm)
#   P2  Risk level traffic light gauges (GREEN/AMBER/RED/CRITICAL)
#   P3  Time-to-threshold countdown bars
#   P4  Compound probability heatmap (country x projection year)
#   P5  Acceleration z-score trend
#   P6  Adaptation measures summary table
#   P7  Markdown recommendation text rendered inline
#
# Inputs   : live_data/merged_dataset.csv          (Cell 34)
#            risk_reports/risk_assessment_latest.json  (Cell 36)
#            risk_reports/recommendations_latest.json  (Cell 37)
#            risk_reports/recommendations_latest.md    (Cell 37)
# Outputs  : dashboards/hypersrl_dashboard_{timestamp}.html
#            dashboards/hypersrl_dashboard_latest.html
#            dashboards/hypersrl_report_{timestamp}.pdf
#            dashboards/hypersrl_report_latest.pdf
#            logs/dashboard_log.csv
# =============================================================================

# --- 0. IMPORTS --------------------------------------------------------------

import subprocess
import sys

def _ensure_package(pkg: str, import_name: str = None) -> None:
    """Install pkg if not already importable. Silent on success."""
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

# Install required packages at module load time so they are available
# for both the test suite (T14) and the main execution path.
_ensure_package("plotly")
_ensure_package("reportlab")

import json
import logging
import textwrap
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell38")

# --- 2. CONFIGURATION --------------------------------------------------------

DRIVE_BASE    = Path("/content/drive/MyDrive/BD_ML_VT")
LIVE_DIR      = DRIVE_BASE / "live_data"
RISK_DIR      = DRIVE_BASE / "risk_reports"
LOG_DIR       = DRIVE_BASE / "logs"
DASH_DIR      = DRIVE_BASE / "dashboards"

MERGED_CSV    = LIVE_DIR / "merged_dataset.csv"
RISK_JSON     = RISK_DIR / "risk_assessment_latest.json"
REC_JSON      = RISK_DIR / "recommendations_latest.json"
REC_MD        = RISK_DIR / "recommendations_latest.md"
DASH_LOG      = LOG_DIR  / "dashboard_log.csv"

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
BASELINE_DATE = pd.Timestamp("2022-12-31")
COUNTRIES     = ["Bangladesh", "Vietnam", "Maldives"]

COUNTRY_COLORS = {
    "Bangladesh": "#1f77b4",
    "Vietnam"   : "#2ca02c",
    "Maldives"  : "#d62728",
}
RISK_COLORS = {
    "GREEN"   : "#2ecc71",
    "AMBER"   : "#f39c12",
    "RED"     : "#e74c3c",
    "CRITICAL": "#8e44ad",
    "UNKNOWN" : "#95a5a6",
}
PROJ_YEARS = [2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080]

SSP245_RATE_MM_YR = {
    "Bangladesh":  5.6,
    "Vietnam"   : 18.5,
    "Maldives"  : 10.2,
}
THRESHOLDS = {
    "Bangladesh": {"amber": 150, "red": 250, "critical": 350},
    "Vietnam"   : {"amber": 100, "red": 200, "critical": 300},
    "Maldives"  : {"amber":  50, "red":  80, "critical": 100},
}

DASH_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- 3. RUNTIME CONTINUITY CHECK ---------------------------------------------

_REQUIRED = {
    "Cell 34 — merged dataset"      : MERGED_CSV,
    "Cell 36 — risk assessment"     : RISK_JSON,
    "Cell 37 — recommendations JSON": REC_JSON,
    "Cell 37 — recommendations MD"  : REC_MD,
}
_missing = [(lbl, p) for lbl, p in _REQUIRED.items() if not Path(p).exists()]
if _missing:
    for lbl, p in _missing:
        logger.error("Missing: [%s] %s", lbl, p)
    raise SystemExit("Re-run Cells 34-37 before Cell 38.")

# --- 4. AUDIT LOG ------------------------------------------------------------

def log_dash_event(stage: str, status: str, detail: str,
                   error: Optional[str] = None) -> None:
    row = pd.DataFrame([{
        "timestamp": RUN_TIMESTAMP, "stage": stage,
        "status": status, "detail": detail,
        "error": error if error else "",
    }])
    if DASH_LOG.exists():
        row.to_csv(DASH_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(DASH_LOG, mode="w", header=True, index=False)
    logger.info("[LOG] %s | %s | %s", stage, status, detail)

# --- 5. DATA LOADING ---------------------------------------------------------

def load_snapshot() -> Dict[str, Any]:
    """Load and freeze all input data into a single snapshot dictionary."""
    logger.info("Loading data snapshot...")

    merged_df  = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    with open(RISK_JSON) as f:
        risk_report = json.load(f)
    with open(REC_JSON) as f:
        recommendations = json.load(f)
    md_text = REC_MD.read_text(encoding="utf-8")

    # Restrict merged_df to post-baseline segment (anomaly-consistent)
    seg_start = BASELINE_DATE - pd.DateOffset(months=18)
    seg_df    = merged_df[merged_df["date"] >= seg_start].copy()

    # Per-country series
    country_series = {}
    for country in COUNTRIES:
        cdf    = seg_df[seg_df["country"] == country].sort_values("date")
        series = cdf.set_index("date")["slrv2"].dropna()
        if series.empty:
            continue
        # Compute cumulative anomaly
        win_start = BASELINE_DATE - pd.DateOffset(months=6)
        win_end   = BASELINE_DATE + pd.DateOffset(months=6)
        win_vals  = series[(series.index >= win_start) &
                           (series.index <= win_end)]
        baseline_val = float(win_vals.mean()) if not win_vals.empty \
                       else float(series.iloc[0])
        cumulative   = series - baseline_val
        country_series[country] = {
            "dates"      : series.index.tolist(),
            "cumulative" : cumulative.tolist(),
            "raw"        : series.tolist(),
        }

    snapshot = {
        "timestamp"      : RUN_TIMESTAMP,
        "baseline_date"  : str(BASELINE_DATE.date()),
        "risk_report"    : risk_report,
        "recommendations": recommendations,
        "md_text"        : md_text,
        "country_series" : country_series,
    }
    logger.info("Snapshot loaded: %d countries with series data",
                len(country_series))
    return snapshot


# =============================================================================
# PANEL BUILDERS — PLOTLY (HTML)
# =============================================================================

def _p1_slr_timeseries(snapshot: Dict) -> "go.Figure":
    """P1: SLR cumulative anomaly time series per country."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=COUNTRIES,
        shared_yaxes=False,
    )

    for col_idx, country in enumerate(COUNTRIES, start=1):
        cs = snapshot["country_series"].get(country)
        r  = snapshot["risk_report"]["countries"].get(country, {})
        t  = r.get("thresholds_mm", THRESHOLDS.get(country, {}))
        color = COUNTRY_COLORS[country]

        if cs:
            dates = [str(d)[:10] for d in cs["dates"]]
            cum   = cs["cumulative"]

            # Main series
            fig.add_trace(go.Scatter(
                x=dates, y=cum,
                mode="lines",
                name=country,
                line=dict(color=color, width=2),
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

            # Threshold lines
            x_range = [dates[0], dates[-1]]
            for level, lcolor, dash in [
                ("amber",    "#f39c12", "dash"),
                ("red",      "#e74c3c", "dot"),
                ("critical", "#8e44ad", "dashdot"),
            ]:
                val = t.get(level)
                if val is not None:
                    fig.add_trace(go.Scatter(
                        x=x_range, y=[val, val],
                        mode="lines",
                        name=f"{level.capitalize()} ({val} mm)",
                        line=dict(color=lcolor, width=1, dash=dash),
                        showlegend=(col_idx == 1),
                    ), row=1, col=col_idx)

            # Baseline marker
            fig.add_vline(
                x=str(BASELINE_DATE.date()),
                line_width=1, line_dash="dot", line_color="gray",
                row=1, col=col_idx,
            )

    fig.update_layout(
        title=dict(
            text="P1 — Cumulative Sea-Level Rise Above 2022-12-31 Baseline",
            font=dict(size=14),
        ),
        height=380,
        margin=dict(l=50, r=20, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
    )
    fig.update_yaxes(title_text="Rise (mm)", col=1)
    fig.update_xaxes(tickangle=30)
    return fig


def _p2_risk_gauges(snapshot: Dict) -> "go.Figure":
    """P2: Risk level traffic light gauges."""
    import plotly.graph_objects as go

    risk_order = ["GREEN", "AMBER", "RED", "CRITICAL"]
    fig        = go.Figure()

    for i, country in enumerate(COUNTRIES):
        r     = snapshot["risk_report"]["countries"].get(country, {})
        level = r.get("risk_level", "UNKNOWN")
        rise  = r.get("current_rise_mm", 0.0)
        t     = r.get("thresholds_mm", THRESHOLDS.get(country, {}))
        crit  = t.get("critical", 350)
        val   = min(max(rise, 0), crit)

        steps = [
            dict(range=[0, t.get("amber", 150)],
                 color=RISK_COLORS["GREEN"]),
            dict(range=[t.get("amber", 150), t.get("red", 250)],
                 color=RISK_COLORS["AMBER"]),
            dict(range=[t.get("red", 250), t.get("critical", 350)],
                 color=RISK_COLORS["RED"]),
            dict(range=[t.get("critical", 350), crit * 1.1],
                 color=RISK_COLORS["CRITICAL"]),
        ]

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=max(rise, 0),
            delta=dict(reference=0, valueformat=".1f",
                       suffix=" mm vs baseline"),
            title=dict(
                text=(f"{country}<br>"
                      f"<span style='font-size:13px;color:"
                      f"{RISK_COLORS[level]}'>"
                      f"<b>{level}</b></span>"),
                font=dict(size=14),
            ),
            number=dict(suffix=" mm", font=dict(size=18)),
            gauge=dict(
                axis=dict(range=[0, crit * 1.1],
                          tickwidth=1),
                bar=dict(color=RISK_COLORS.get(level, "#95a5a6"),
                         thickness=0.25),
                steps=steps,
                threshold=dict(
                    line=dict(color="black", width=3),
                    thickness=0.75,
                    value=t.get("amber", 150),
                ),
            ),
            domain=dict(
                x=[i / 3 + 0.02, (i + 1) / 3 - 0.02],
                y=[0, 1],
            ),
        ))

    fig.update_layout(
        title=dict(
            text="P2 — Risk Level Gauges (mm above 2022 baseline)",
            font=dict(size=14),
        ),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
    )
    return fig


def _p3_threshold_countdown(snapshot: Dict) -> "go.Figure":
    """P3: Time-to-threshold countdown horizontal bars."""
    import plotly.graph_objects as go

    levels  = ["amber", "red", "critical"]
    fig     = go.Figure()
    y_ticks = []
    y_pos   = 0

    for country in COUNTRIES:
        r = snapshot["risk_report"]["countries"].get(country, {})
        t = r.get("time_to_threshold", {})
        for level in levels:
            yrs = t.get(f"years_to_{level}", None)
            if yrs is None:
                continue
            label  = f"{country} — {level.upper()}"
            color  = RISK_COLORS.get(level.upper(),
                                     RISK_COLORS.get("AMBER"))
            if yrs == 0:
                bar_val  = 0.3
                bar_text = "CROSSED"
                bar_col  = "#bdc3c7"
            else:
                bar_val  = min(float(yrs), 80)
                bar_text = f"{yrs:.1f} yr"
                bar_col  = color

            fig.add_trace(go.Bar(
                x=[bar_val],
                y=[label],
                orientation="h",
                marker_color=bar_col,
                text=bar_text,
                textposition="outside",
                showlegend=False,
                width=0.6,
            ))
            y_ticks.append(label)
            y_pos += 1

    fig.update_layout(
        title=dict(
            text="P3 — Years to Risk Threshold (SSP2-4.5 projection)",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Years from now",
            range=[0, 85],
        ),
        height=420,
        margin=dict(l=200, r=80, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=10),
        bargap=0.3,
    )
    return fig


def _p4_compound_heatmap(snapshot: Dict) -> "go.Figure":
    """P4: Compound event probability heatmap (country x projection year)."""
    import plotly.graph_objects as go
    from scipy.special import ndtr

    rows  = []
    for country in COUNTRIES:
        r    = snapshot["risk_report"]["countries"].get(country, {})
        rise = r.get("current_rise_mm", 0.0)
        rate = SSP245_RATE_MM_YR.get(country, 6.0)
        thr  = THRESHOLDS[country]["red"]
        srp  = {"Bangladesh": 10, "Vietnam": 10, "Maldives": 5}[country]
        surge_prob = 1.0 / srp

        row = []
        for yr in PROJ_YEARS:
            years_fwd   = max(yr - 2024, 1)
            proj_slr    = rise + rate * years_fwd
            slr_sigma   = max(abs(proj_slr) * 0.20, 10.0)
            slr_exceed  = float(np.clip(
                1.0 - ndtr((thr - proj_slr) / slr_sigma), 0, 1
            ))
            theta = 2.0
            u = max(slr_exceed,  1e-10)
            v = max(surge_prob,  1e-10)
            cop = float(np.exp(
                -((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)
            ))
            row.append(round(min(cop, 1.0) * 100, 2))
        rows.append(row)

    z_text = [[f"{v:.1f}%" for v in row] for row in rows]

    fig = go.Figure(go.Heatmap(
        z=rows,
        x=[str(y) for y in PROJ_YEARS],
        y=COUNTRIES,
        text=z_text,
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#2ecc71"],
            [0.05, "#f1c40f"],
            [0.15, "#e67e22"],
            [0.30, "#e74c3c"],
            [1.0,  "#8e44ad"],
        ],
        colorbar=dict(title="Annual prob (%)"),
        zmin=0, zmax=30,
    ))

    fig.update_layout(
        title=dict(
            text="P4 — Compound SLR + Storm Surge Annual Probability (%)",
            font=dict(size=14),
        ),
        xaxis_title="Projection year",
        height=280,
        margin=dict(l=100, r=20, t=50, b=50),
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
    )
    return fig


def _p5_acceleration(snapshot: Dict) -> "go.Figure":
    """P5: Acceleration z-score per country (bar chart with alert line)."""
    import plotly.graph_objects as go

    zscores = []
    labels  = []
    colors  = []
    notes   = []

    for country in COUNTRIES:
        r    = snapshot["risk_report"]["countries"].get(country, {})
        acc  = r.get("acceleration", {})
        z    = acc.get("z_score", 0.0) or 0.0
        det  = acc.get("acceleration_detected", False)
        note = acc.get("note", "")
        zscores.append(z)
        labels.append(country)
        colors.append(RISK_COLORS["RED"] if det else COUNTRY_COLORS[country])
        notes.append(note or "")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=zscores,
        marker_color=colors,
        text=[f"z = {z:.3f}" for z in zscores],
        textposition="outside",
        name="Z-score",
    ))

    # Alert threshold lines
    for sign, label in [(2.0, "Alert (+2σ)"), (-2.0, "Alert (−2σ)")]:
        fig.add_hline(
            y=sign,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=label,
            annotation_position="right",
        )

    # Hover notes
    for i, country in enumerate(COUNTRIES):
        if notes[i]:
            fig.add_annotation(
                x=country, y=zscores[i],
                text=f"<i>{notes[i][:40]}</i>",
                showarrow=False,
                yshift=-30,
                font=dict(size=9, color="gray"),
            )

    fig.update_layout(
        title=dict(
            text="P5 — Sea-Level Acceleration Z-Score (24-month rolling trend)",
            font=dict(size=14),
        ),
        yaxis=dict(title="Z-score", zeroline=True,
                   zerolinecolor="black", zerolinewidth=1),
        height=300,
        margin=dict(l=60, r=80, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        showlegend=False,
    )
    return fig


def _p6_measures_table(snapshot: Dict) -> "go.Figure":
    """P6: Adaptation measures summary table."""
    import plotly.graph_objects as go

    adp      = snapshot["recommendations"].get(
        "five_year_adaptation_measures", {}
    )
    headers  = ["Country", "Measure", "Priority", "Cost", "Timeline"]
    rows     = [[], [], [], [], []]

    priority_colors = {
        "HIGH"  : "#e74c3c",
        "MEDIUM": "#f39c12",
        "LOW"   : "#2ecc71",
    }

    for section in [*COUNTRIES, "regional"]:
        label    = section if section != "regional" else "Regional"
        measures = adp.get(section, [])
        for m in measures:
            rows[0].append(label)
            rows[1].append(
                textwrap.shorten(m.get("measure", ""), width=48, placeholder="…")
            )
            rows[2].append(m.get("priority",  ""))
            rows[3].append(m.get("cost_tier", ""))
            rows[4].append(m.get("timeline",  ""))

    # Color priority column
    priority_fill = [
        priority_colors.get(p, "#ecf0f1") for p in rows[2]
    ]

    fill_colors = [
        ["#f8f9fa"] * len(rows[0]),   # Country
        ["white"] * len(rows[0]),      # Measure
        priority_fill,                 # Priority
        ["white"] * len(rows[0]),      # Cost
        ["white"] * len(rows[0]),      # Timeline
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#2c3e50",
            font=dict(color="white", size=11),
            align="left",
            height=28,
        ),
        cells=dict(
            values=rows,
            fill_color=fill_colors,
            font=dict(size=10),
            align="left",
            height=22,
        ),
    ))

    fig.update_layout(
        title=dict(
            text="P6 — 5-Year Adaptation Measures Summary",
            font=dict(size=14),
        ),
        height=max(350, len(rows[0]) * 22 + 100),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
    )
    return fig


def _p7_recommendations_html(snapshot: Dict) -> str:
    """P7: Render the Markdown recommendation report as styled HTML block."""
    import re

    md = snapshot["md_text"]

    # Minimal Markdown -> HTML conversion (no external library needed)
    lines   = md.split("\n")
    html    = []
    in_table = False
    in_list  = False

    for line in lines:
        # Tables
        if line.startswith("|"):
            if not in_table:
                html.append('<table class="md-table">')
                in_table = True
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Header separator row
            if all(set(c.replace("-","").replace(":","").strip()) <= {""} or
                   c.strip().replace("-","").replace(":","") == ""
                   for c in cells):
                continue
            tag = "th" if not any(
                '<th>' in h for h in html[-3:]
            ) else "td"
            # Detect if previous row was header
            if html and "<th>" not in html[-1] and "<td>" not in html[-1]:
                tag = "th"
            else:
                tag = "td"
            row_html = "".join(
                f"<{tag}>{c}</{tag}>" for c in cells
            )
            html.append(f"<tr>{row_html}</tr>")
            continue
        else:
            if in_table:
                html.append("</table>")
                in_table = False

        # Lists
        if line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            content = line[2:]
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            html.append(f"<li>{content}</li>")
            continue
        else:
            if in_list:
                html.append("</ul>")
                in_list = False

        # Headings
        if line.startswith("#### "):
            text = line[5:]
            html.append(f"<h4>{text}</h4>")
        elif line.startswith("### "):
            text = line[4:]
            html.append(f"<h3>{text}</h3>")
        elif line.startswith("## "):
            text = line[3:]
            html.append(f"<h2>{text}</h2>")
        elif line.startswith("# "):
            text = line[2:]
            html.append(f"<h1>{text}</h1>")
        elif line.strip() == "---":
            html.append("<hr>")
        elif line.strip() == "":
            html.append("<br>")
        else:
            text = line
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"\*(.+?)\*",   r"<em>\1</em>",           text)
            text = re.sub(r"`(.+?)`",     r"<code>\1</code>",        text)
            html.append(f"<p>{text}</p>")

    if in_table: html.append("</table>")
    if in_list:  html.append("</ul>")

    return "\n".join(html)


# =============================================================================
# HTML DASHBOARD BUILDER
# =============================================================================

def build_html_dashboard(snapshot: Dict) -> str:
    """
    Assemble all 7 panels into a single self-contained HTML file.
    All Plotly figures are embedded as JSON (no CDN dependency after load).
    """
    import plotly.io as pio

    logger.info("Building HTML dashboard...")

    figures = {
        "p1": _p1_slr_timeseries(snapshot),
        "p2": _p2_risk_gauges(snapshot),
        "p3": _p3_threshold_countdown(snapshot),
        "p4": _p4_compound_heatmap(snapshot),
        "p5": _p5_acceleration(snapshot),
        "p6": _p6_measures_table(snapshot),
    }

    # Serialize each figure to div (includes plotly.js inline for first)
    divs     = {}
    include_js = True
    for key, fig in figures.items():
        divs[key] = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=include_js,
            config={"displayModeBar": True, "responsive": True},
        )
        include_js = False   # Only embed plotly.js once

    p7_html = _p7_recommendations_html(snapshot)

    rpt    = snapshot["risk_report"]
    rec    = snapshot["recommendations"]
    meta   = rec.get("report_metadata", {})
    sys_risk = rpt.get("system_risk_level", "UNKNOWN")
    risk_color = RISK_COLORS.get(sys_risk, "#95a5a6")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HyperSLR Digital Twin Dashboard — {RUN_TIMESTAMP}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Arial', sans-serif;
      background: #f0f2f5;
      color: #2c3e50;
    }}
    .header {{
      background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%);
      color: white;
      padding: 24px 32px;
      border-bottom: 4px solid {risk_color};
    }}
    .header h1 {{
      font-size: 22px;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .header .meta {{
      font-size: 12px;
      opacity: 0.75;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .risk-badge {{
      display: inline-block;
      padding: 4px 14px;
      border-radius: 20px;
      font-weight: bold;
      font-size: 13px;
      background: {risk_color};
      color: white;
      margin-left: 10px;
      vertical-align: middle;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 20px;
    }}
    .panel {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      margin-bottom: 24px;
      overflow: hidden;
    }}
    .panel-inner {{
      padding: 4px 8px 8px 8px;
    }}
    .panel-label {{
      font-size: 10px;
      font-weight: bold;
      color: #7f8c8d;
      letter-spacing: 1px;
      text-transform: uppercase;
      padding: 8px 16px 0 16px;
    }}
    .section-title {{
      font-size: 16px;
      font-weight: bold;
      color: #2c3e50;
      padding: 20px 16px 8px 16px;
      border-top: 3px solid #3498db;
      margin-top: 8px;
    }}
    /* Recommendation panel */
    .rec-panel {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      padding: 24px 32px;
      margin-bottom: 24px;
    }}
    .rec-panel h1 {{
      font-size: 20px;
      color: #2c3e50;
      margin: 16px 0 8px 0;
      border-bottom: 2px solid #3498db;
      padding-bottom: 6px;
    }}
    .rec-panel h2 {{
      font-size: 16px;
      color: #2980b9;
      margin: 20px 0 8px 0;
    }}
    .rec-panel h3 {{
      font-size: 14px;
      color: #34495e;
      margin: 14px 0 6px 0;
    }}
    .rec-panel h4 {{
      font-size: 13px;
      color: #7f8c8d;
      margin: 10px 0 4px 0;
    }}
    .rec-panel p {{
      font-size: 12px;
      line-height: 1.7;
      margin-bottom: 6px;
      color: #34495e;
    }}
    .rec-panel ul {{
      padding-left: 20px;
      margin-bottom: 8px;
    }}
    .rec-panel li {{
      font-size: 12px;
      line-height: 1.6;
      color: #34495e;
      margin-bottom: 2px;
    }}
    .rec-panel hr {{
      border: none;
      border-top: 1px solid #ecf0f1;
      margin: 16px 0;
    }}
    .rec-panel code {{
      background: #ecf0f1;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 11px;
    }}
    .md-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      margin: 8px 0 12px 0;
    }}
    .md-table th {{
      background: #2c3e50;
      color: white;
      padding: 6px 10px;
      text-align: left;
    }}
    .md-table td {{
      padding: 5px 10px;
      border-bottom: 1px solid #ecf0f1;
    }}
    .md-table tr:nth-child(even) td {{ background: #f8f9fa; }}
    .footer {{
      text-align: center;
      font-size: 11px;
      color: #95a5a6;
      padding: 16px;
      margin-top: 8px;
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>
    HyperSLR Digital Twin — Coastal Flood Risk Dashboard
    <span class="risk-badge">{sys_risk}</span>
  </h1>
  <div class="meta">
    <span>Generated: {RUN_TIMESTAMP[:8][:4]}-{RUN_TIMESTAMP[4:6]}-{RUN_TIMESTAMP[6:8]} {RUN_TIMESTAMP[9:11]}:{RUN_TIMESTAMP[11:13]} UTC</span>
    <span>Baseline: {snapshot['baseline_date']}</span>
    <span>LLM provider: {meta.get('generated_by', 'N/A')}</span>
    <span>Framework: HyperSLR Cell 38</span>
  </div>
</div>

<div class="container">

  <div class="panel">
    <div class="panel-label">Panel 1 — Sea-Level Rise Time Series</div>
    <div class="panel-inner">{divs['p1']}</div>
  </div>

  <div class="panel">
    <div class="panel-label">Panel 2 — Risk Level Gauges</div>
    <div class="panel-inner">{divs['p2']}</div>
  </div>

  <div class="panel">
    <div class="panel-label">Panel 3 — Time to Risk Threshold</div>
    <div class="panel-inner">{divs['p3']}</div>
  </div>

  <div class="panel">
    <div class="panel-label">Panel 4 — Compound Event Probability Heatmap</div>
    <div class="panel-inner">{divs['p4']}</div>
  </div>

  <div class="panel">
    <div class="panel-label">Panel 5 — Acceleration Z-Score</div>
    <div class="panel-inner">{divs['p5']}</div>
  </div>

  <div class="panel">
    <div class="panel-label">Panel 6 — Adaptation Measures Table</div>
    <div class="panel-inner">{divs['p6']}</div>
  </div>

  <div class="rec-panel">
    <div class="panel-label" style="padding:0 0 8px 0;">
      Panel 7 — Adaptation Recommendations Report
    </div>
    {p7_html}
  </div>

</div>

<div class="footer">
  HyperSLR Digital Twin Framework &mdash; Cell 38 Dashboard &mdash;
  Risk data: Cell 36 &mdash; Recommendations: Cell 37
  &mdash; Snapshot frozen at {RUN_TIMESTAMP}
</div>

</body>
</html>"""

    log_dash_event("HTML", "SUCCESS",
                   f"panels=7, size={len(html)//1024}KB")
    return html


# =============================================================================
# PDF REPORT BUILDER
# =============================================================================

def build_pdf_report(snapshot: Dict, output_path: Path) -> None:
    """
    Build a publication-quality PDF report using ReportLab + Matplotlib.

    Structure:
      Page 1  : Cover — title, system risk, country status table
      Page 2  : P1 SLR time series (Matplotlib figure)
      Page 3  : P2 Gauges + P3 Threshold bars (Matplotlib)
      Page 4  : P4 Compound heatmap + P5 Acceleration (Matplotlib)
      Page 5  : P6 Adaptation measures table (ReportLab table)
      Page 6+ : P7 Recommendations text
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage, HRFlowable,
    )

    logger.info("Building PDF report...")

    rpt      = snapshot["risk_report"]
    rec      = snapshot["recommendations"]
    meta_rec = rec.get("report_metadata", {})
    sys_risk = rpt.get("system_risk_level", "UNKNOWN")
    adp      = rec.get("five_year_adaptation_measures", {})
    pol      = rec.get("policy_recommendations", {})
    trigs    = rec.get("monitoring_triggers", [])
    uncerts  = rec.get("critical_uncertainties", [])

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    W, H   = A4

    # Custom styles
    title_style = ParagraphStyle(
        "HTitle",
        parent=styles["Title"],
        fontSize=18, spaceAfter=6, leading=22,
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=13, spaceAfter=4, spaceBefore=10,
        textColor=rl_colors.HexColor("#2c3e50"),
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=11, spaceAfter=3, spaceBefore=8,
        textColor=rl_colors.HexColor("#2980b9"),
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, spaceAfter=3, leading=13,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=8, spaceAfter=2, leading=11,
        textColor=rl_colors.HexColor("#555555"),
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, spaceAfter=4, leading=10,
        textColor=rl_colors.HexColor("#7f8c8d"),
        alignment=TA_CENTER,
    )

    def _risk_color_rl(level: str):
        hex_map = {
            "GREEN"   : "#2ecc71",
            "AMBER"   : "#f39c12",
            "RED"     : "#e74c3c",
            "CRITICAL": "#8e44ad",
            "UNKNOWN" : "#95a5a6",
        }
        return rl_colors.HexColor(hex_map.get(level, "#95a5a6"))

    def _fig_to_image(fig, width_cm=16.5, height_cm=8) -> RLImage:
        """Convert a Matplotlib figure to a ReportLab Image flowable."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        plt.close(fig)
        return RLImage(buf, width=width_cm*cm, height=height_cm*cm)

    story = []

    # ------------------------------------------------------------------
    # COVER PAGE
    # ------------------------------------------------------------------
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph(
        "HyperSLR Digital Twin Framework",
        ParagraphStyle("Cover1", parent=styles["Normal"],
                       fontSize=11, textColor=rl_colors.HexColor("#7f8c8d"),
                       alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Coastal Flood Risk Assessment Report",
        ParagraphStyle("Cover2", parent=styles["Title"],
                       fontSize=20, alignment=TA_CENTER,
                       textColor=rl_colors.HexColor("#2c3e50"))
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Bangladesh &nbsp;|&nbsp; Vietnam &nbsp;|&nbsp; Maldives",
        ParagraphStyle("Cover3", parent=styles["Normal"],
                       fontSize=12, alignment=TA_CENTER,
                       textColor=rl_colors.HexColor("#7f8c8d"))
    ))
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=_risk_color_rl(sys_risk)))
    story.append(Spacer(1, 0.4*cm))

    # Meta table
    gen_ts = meta_rec.get("generation_ts", RUN_TIMESTAMP)[:19].replace("T", " ")
    meta_data = [
        ["Report date",       gen_ts + " UTC"],
        ["Baseline date",     snapshot["baseline_date"]],
        ["System risk level", sys_risk],
        ["LLM provider",      meta_rec.get("generated_by", "N/A")],
        ["Framework version", "HyperSLR Cell 38"],
        ["Audience",          "National coastal adaptation planners"],
    ]
    meta_table = Table(meta_data, colWidths=[4.5*cm, 11*cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), rl_colors.HexColor("#2c3e50")),
        ("BACKGROUND",  (0, 0), (-1, -1), rl_colors.HexColor("#f8f9fa")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [rl_colors.HexColor("#f8f9fa"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#ecf0f1")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story += [meta_table, Spacer(1, 0.6*cm)]

    # Country status table
    story.append(Paragraph("Country Risk Status", h1_style))
    status_header = [
        [Paragraph("<b>Country</b>",       small_style),
         Paragraph("<b>Risk Level</b>",    small_style),
         Paragraph("<b>Rise (mm)</b>",     small_style),
         Paragraph("<b>Next Threshold</b>",small_style),
         Paragraph("<b>Years</b>",         small_style),
         Paragraph("<b>Compound Prob</b>", small_style)],
    ]
    status_rows = []
    for country in COUNTRIES:
        r     = rpt["countries"].get(country, {})
        level = r.get("risk_level", "N/A")
        rise  = r.get("current_rise_mm", 0.0)
        t     = r.get("time_to_threshold", {})
        cmp   = r.get("compound_risk", {})
        status_rows.append([
            Paragraph(country, small_style),
            Paragraph(f"<b>{level}</b>", small_style),
            Paragraph(f"{rise:.1f}", small_style),
            Paragraph(
                str(t.get("next_threshold","N/A")).upper(), small_style
            ),
            Paragraph(str(t.get("years_to_next","N/A")), small_style),
            Paragraph(
                f"{cmp.get('compound_prob_annual',0):.4f}", small_style
            ),
        ])

    status_table = Table(
        status_header + status_rows,
        colWidths=[3.5*cm, 2.5*cm, 2*cm, 3*cm, 1.8*cm, 2.7*cm],
    )
    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), rl_colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [rl_colors.white, rl_colors.HexColor("#f8f9fa")]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ])
    for i, country in enumerate(COUNTRIES, start=1):
        level = rpt["countries"].get(country, {}).get("risk_level", "UNKNOWN")
        ts.add("BACKGROUND", (1, i), (1, i), _risk_color_rl(level))
        ts.add("TEXTCOLOR",  (1, i), (1, i), rl_colors.white)
    status_table.setStyle(ts)
    story += [status_table, PageBreak()]

    # ------------------------------------------------------------------
    # P1: SLR Time Series (Matplotlib)
    # ------------------------------------------------------------------
    story.append(Paragraph("Figure 1 — Cumulative Sea-Level Rise", h1_style))
    story.append(Paragraph(
        "Cumulative SLR anomaly (mm) above the 2022-12-31 baseline. "
        "Dashed/dotted horizontal lines indicate AMBER, RED, and CRITICAL "
        "risk thresholds. Grey vertical line marks the baseline date.",
        caption_style,
    ))
    story.append(Spacer(1, 0.2*cm))

    fig1, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig1.patch.set_facecolor("white")
    line_styles = {
        "amber"   : dict(color="#f39c12", lw=1.2, ls="--",  label="Amber"),
        "red"     : dict(color="#e74c3c", lw=1.2, ls=":",   label="Red"),
        "critical": dict(color="#8e44ad", lw=1.2, ls="-.",  label="Critical"),
    }
    for ax, country in zip(axes, COUNTRIES):
        cs  = snapshot["country_series"].get(country)
        thr = THRESHOLDS[country]
        ax.set_facecolor("white")
        ax.set_title(country, fontsize=11, fontweight="bold")
        if cs:
            dates = pd.DatetimeIndex(cs["dates"])
            cum   = np.array(cs["cumulative"])
            ax.plot(dates, cum,
                    color=COUNTRY_COLORS[country], lw=2, label="Observed")
            for level, sty in line_styles.items():
                ax.axhline(thr[level], **sty)
        ax.axvline(BASELINE_DATE, color="gray", lw=1, ls=":")
        ax.set_xlabel("", fontsize=8)
        ax.set_ylabel("Rise (mm)" if country == "Bangladesh" else "")
        ax.tick_params(labelsize=8)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    handles = [mpatches.Patch(color=COUNTRY_COLORS[c], label=c)
               for c in COUNTRIES]
    for level, sty in line_styles.items():
        handles.append(plt.Line2D([0],[0], **{k:v for k,v in sty.items()
                                              if k != "label"},
                                  label=sty["label"]))
    fig1.legend(handles=handles, loc="lower center",
                ncol=6, fontsize=8, frameon=False,
                bbox_to_anchor=(0.5, -0.08))
    fig1.tight_layout()
    story.append(_fig_to_image(fig1, height_cm=9))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # P2 + P3 on one page
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "Figure 2 — Risk Gauges and Threshold Timeline", h1_style
    ))

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={"polar": False})
    fig2.patch.set_facecolor("white")
    for ax2, country in zip(axes2, COUNTRIES):
        r     = rpt["countries"].get(country, {})
        level = r.get("risk_level", "UNKNOWN")
        rise  = max(r.get("current_rise_mm", 0), 0)
        thr   = THRESHOLDS[country]
        crit  = thr["critical"]

        wedge_data = [
            (thr["amber"],            RISK_COLORS["GREEN"]),
            (thr["red"]-thr["amber"], RISK_COLORS["AMBER"]),
            (crit-thr["red"],         RISK_COLORS["RED"]),
            (crit*0.2,                RISK_COLORS["CRITICAL"]),
        ]
        sizes = [w[0] for w in wedge_data]
        clrs  = [w[1] for w in wedge_data]
        wedges, _ = ax2.pie(
            sizes, colors=clrs, startangle=180,
            counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor="white"),
        )
        # Needle
        needle_frac = min(rise / (crit * 1.2), 1.0)
        angle       = 180 - needle_frac * 180
        angle_rad   = np.radians(angle)
        ax2.annotate("", xy=(0.6*np.cos(angle_rad), 0.6*np.sin(angle_rad)),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax2.text(0, -0.35, f"{rise:.1f} mm",
                 ha="center", va="center", fontsize=11, fontweight="bold")
        ax2.text(0, -0.55, level,
                 ha="center", va="center", fontsize=10,
                 color=RISK_COLORS.get(level, "#95a5a6"), fontweight="bold")
        ax2.set_title(country, fontsize=11, fontweight="bold", pad=10)
        ax2.set(aspect="equal")
        ax2.axis("off")
    fig2.tight_layout()
    story.append(_fig_to_image(fig2, height_cm=7))
    story.append(Spacer(1, 0.3*cm))

    # P3 — threshold bars
    story.append(Paragraph(
        "Figure 3 — Years to Risk Threshold (SSP2-4.5)", h2_style
    ))
    bar_labels = []
    bar_vals   = []
    bar_colors = []
    for country in COUNTRIES:
        r = rpt["countries"].get(country, {})
        t = r.get("time_to_threshold", {})
        for level in ["amber", "red", "critical"]:
            yrs = t.get(f"years_to_{level}", None)
            if yrs is None: continue
            bar_labels.append(f"{country}\n{level.upper()}")
            bar_vals.append(min(float(yrs) if yrs > 0 else 0.3, 80))
            bar_colors.append(
                "#bdc3c7" if yrs == 0 else RISK_COLORS.get(level.upper(), "#3498db")
            )
    fig3, ax3 = plt.subplots(figsize=(14, 3.5))
    fig3.patch.set_facecolor("white")
    ax3.set_facecolor("white")
    y_pos = range(len(bar_labels))
    ax3.barh(list(y_pos), bar_vals, color=bar_colors, height=0.6)
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels(bar_labels, fontsize=8)
    ax3.set_xlabel("Years from now (SSP2-4.5)", fontsize=9)
    ax3.set_xlim(0, 85)
    ax3.axvline(0, color="black", lw=0.5)
    ax3.grid(axis="x", alpha=0.3)
    ax3.spines[["top","right"]].set_visible(False)
    for i, (v, lbl) in enumerate(zip(bar_vals, bar_labels)):
        txt = "CROSSED" if v <= 0.35 else f"{v:.1f} yr"
        ax3.text(v + 0.5, i, txt, va="center", fontsize=8)
    fig3.tight_layout()
    story.append(_fig_to_image(fig3, height_cm=5.5))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # P4 + P5 on one page
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "Figure 4 — Compound Event Probability Heatmap", h1_style
    ))
    story.append(Paragraph(
        "Annual probability (%) of compound SLR exceedance + storm surge "
        "event, estimated via Gumbel copula (theta=2.0, SSP2-4.5). "
        "Red cell = exceedance probable in that decade.",
        caption_style
    ))
    from scipy.special import ndtr as _ndtr
    z_matrix = []
    for country in COUNTRIES:
        r    = rpt["countries"].get(country, {})
        rise = r.get("current_rise_mm", 0.0)
        rate = SSP245_RATE_MM_YR.get(country, 6.0)
        thr  = THRESHOLDS[country]["red"]
        srp  = {"Bangladesh": 10, "Vietnam": 10, "Maldives": 5}[country]
        row  = []
        for yr in PROJ_YEARS:
            proj   = rise + rate * max(yr - 2024, 1)
            sigma  = max(abs(proj) * 0.2, 10.0)
            sl_ex  = float(np.clip(1 - _ndtr((thr - proj)/sigma), 0, 1))
            u, v   = max(sl_ex, 1e-10), max(1/srp, 1e-10)
            cop    = float(np.exp(
                -((-np.log(u))**2 + (-np.log(v))**2)**0.5
            ))
            row.append(round(min(cop, 1.0) * 100, 2))
        z_matrix.append(row)

    fig4, ax4 = plt.subplots(figsize=(12, 3))
    fig4.patch.set_facecolor("white")
    cmap4 = mcolors.LinearSegmentedColormap.from_list(
        "slr", ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"]
    )
    im = ax4.imshow(z_matrix, cmap=cmap4, vmin=0, vmax=30, aspect="auto")
    ax4.set_xticks(range(len(PROJ_YEARS)))
    ax4.set_xticklabels([str(y) for y in PROJ_YEARS], fontsize=9)
    ax4.set_yticks(range(len(COUNTRIES)))
    ax4.set_yticklabels(COUNTRIES, fontsize=9)
    for i in range(len(COUNTRIES)):
        for j in range(len(PROJ_YEARS)):
            ax4.text(j, i, f"{z_matrix[i][j]:.1f}%",
                     ha="center", va="center", fontsize=8,
                     color="white" if z_matrix[i][j] > 12 else "black")
    plt.colorbar(im, ax=ax4, label="Annual prob (%)", shrink=0.8)
    ax4.set_title("Compound SLR + Storm Surge Annual Probability (%)",
                  fontsize=10)
    fig4.tight_layout()
    story.append(_fig_to_image(fig4, height_cm=5))
    story.append(Spacer(1, 0.4*cm))

    # P5 — acceleration
    story.append(Paragraph(
        "Figure 5 — Acceleration Z-Score (24-month rolling)", h2_style
    ))
    acc_countries = []
    acc_zscores   = []
    acc_colors    = []
    for country in COUNTRIES:
        r   = rpt["countries"].get(country, {})
        acc = r.get("acceleration", {})
        z   = acc.get("z_score", 0.0) or 0.0
        det = acc.get("acceleration_detected", False)
        acc_countries.append(country)
        acc_zscores.append(z)
        acc_colors.append(RISK_COLORS["RED"] if det
                          else COUNTRY_COLORS[country])

    fig5, ax5 = plt.subplots(figsize=(8, 3))
    fig5.patch.set_facecolor("white")
    ax5.set_facecolor("white")
    bars = ax5.bar(acc_countries, acc_zscores,
                   color=acc_colors, width=0.4, zorder=3)
    ax5.axhline( 2.0, color="#e74c3c", ls="--", lw=1.2,
                 label="Alert threshold (±2σ)")
    ax5.axhline(-2.0, color="#e74c3c", ls="--", lw=1.2)
    ax5.axhline( 0,   color="black",   ls="-",  lw=0.5)
    for bar, z in zip(bars, acc_zscores):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 z + (0.05 if z >= 0 else -0.15),
                 f"z={z:.3f}", ha="center", fontsize=9)
    ax5.set_ylabel("Z-score", fontsize=9)
    ax5.legend(fontsize=8, frameon=False)
    ax5.grid(axis="y", alpha=0.3, zorder=0)
    ax5.spines[["top","right"]].set_visible(False)
    fig5.tight_layout()
    story.append(_fig_to_image(fig5, height_cm=4.5))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # P6: Adaptation Measures Table
    # ------------------------------------------------------------------
    story.append(Paragraph("Table 1 — 5-Year Adaptation Measures", h1_style))

    tbl_header = [
        [Paragraph(h, ParagraphStyle("TH", parent=small_style,
                                     fontName="Helvetica-Bold",
                                     textColor=rl_colors.white))
         for h in ["Country", "Measure", "Priority", "Cost", "Timeline"]]
    ]
    tbl_rows = []
    for section in [*COUNTRIES, "regional"]:
        label    = section if section != "regional" else "Regional"
        measures = adp.get(section, [])
        for m in measures:
            tbl_rows.append([
                Paragraph(label, small_style),
                Paragraph(
                    textwrap.shorten(m.get("measure",""), 52, placeholder="…"),
                    small_style
                ),
                Paragraph(m.get("priority",  ""), small_style),
                Paragraph(m.get("cost_tier", ""), small_style),
                Paragraph(m.get("timeline",  ""), small_style),
            ])

    p_colors_rl = {
        "HIGH"  : rl_colors.HexColor("#e74c3c"),
        "MEDIUM": rl_colors.HexColor("#f39c12"),
        "LOW"   : rl_colors.HexColor("#2ecc71"),
    }

    tbl = Table(
        tbl_header + tbl_rows,
        colWidths=[2.5*cm, 8*cm, 1.8*cm, 1.8*cm, 2.4*cm],
        repeatRows=1,
    )
    ts2 = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), rl_colors.HexColor("#2c3e50")),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [rl_colors.white, rl_colors.HexColor("#f8f9fa")]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ])
    for i, row in enumerate(tbl_rows, start=1):
        pri_text = adp.get(
            list(adp.keys())[0], [{}]
        )
        # Color priority cell
        raw_rows_flat = []
        for section in [*COUNTRIES, "regional"]:
            for m in adp.get(section, []):
                raw_rows_flat.append(m.get("priority", ""))
        if i - 1 < len(raw_rows_flat):
            pri = raw_rows_flat[i-1]
            if pri in p_colors_rl:
                ts2.add("BACKGROUND", (2, i), (2, i), p_colors_rl[pri])
                ts2.add("TEXTCOLOR",  (2, i), (2, i), rl_colors.white)
    tbl.setStyle(ts2)
    story += [tbl, PageBreak()]

    # ------------------------------------------------------------------
    # P7: Recommendations Text
    # ------------------------------------------------------------------
    story.append(Paragraph("Adaptation Recommendations", h1_style))
    story.append(Paragraph(
        f"Generated by: {meta_rec.get('generated_by','N/A')} &nbsp;|&nbsp; "
        f"Audience: National coastal adaptation planners",
        small_style,
    ))
    story.append(Spacer(1, 0.3*cm))

    def _add_section(title: str, items: List[Dict],
                     key_field: str, rationale_field: str,
                     sub_list_field: str, sub_list_label: str) -> None:
        story.append(Paragraph(title, h2_style))
        for item in items:
            heading = item.get(key_field, "")
            rat     = item.get(rationale_field, "")
            subs    = item.get(sub_list_field, [])
            extras  = {k: v for k, v in item.items()
                       if k not in [key_field, rationale_field,
                                    sub_list_field, "country"]}
            meta_str = "  |  ".join(
                f"{k.replace('_',' ').title()}: {v}"
                for k, v in extras.items()
                if isinstance(v, str) and v
            )
            story.append(Paragraph(f"<b>{heading}</b>", body_style))
            if meta_str:
                story.append(Paragraph(
                    f"<i>{meta_str}</i>", small_style
                ))
            if rat:
                story.append(Paragraph(rat, body_style))
            if subs:
                story.append(Paragraph(
                    f"<b>{sub_list_label}:</b>", small_style
                ))
                for s in subs:
                    story.append(Paragraph(f"• {s}", small_style))
            story.append(Spacer(1, 0.15*cm))

    for country in [*COUNTRIES, "regional"]:
        label = country if country != "regional" else "Regional / Cross-Border"
        story.append(Paragraph(label, h1_style))

        country_adp = adp.get(country, [])
        if country_adp:
            _add_section(
                "5-Year Adaptation Measures", country_adp,
                "measure", "rationale", "actions", "Actions"
            )

        country_pol = pol.get(country, [])
        if country_pol:
            _add_section(
                "Policy Recommendations", country_pol,
                "policy", "rationale", "instruments", "Instruments"
            )

    # Monitoring triggers
    story.append(Paragraph("Monitoring Triggers", h1_style))
    for trig in trigs:
        story.append(Paragraph(
            f"<b>{trig.get('trigger','')}</b> — "
            f"Threshold: {trig.get('threshold','')}",
            body_style,
        ))
        story.append(Paragraph(
            f"Response: {trig.get('response_action','')}", small_style
        ))
        story.append(Spacer(1, 0.1*cm))

    # Critical uncertainties
    story.append(Paragraph("Critical Uncertainties", h1_style))
    for u in uncerts:
        story.append(Paragraph(f"• {u}", small_style))

    doc.build(story)
    logger.info("PDF built: %s", output_path)
    log_dash_event("PDF", "SUCCESS", f"path={output_path.name}")


# =============================================================================
# CELL 38 — MAIN EXECUTION
# =============================================================================

def run_cell_38() -> Dict[str, Any]:
    """Execute the full Cell 38 dashboard and PDF pipeline."""
    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 38: DASHBOARD AND PUBLICATION REPORT")
    logger.info("Run timestamp: %s", RUN_TIMESTAMP)
    logger.info("=" * 70)

    snapshot = load_snapshot()

    # --- HTML Dashboard ---
    logger.info("--- BUILDING HTML DASHBOARD ---")
    html_content = build_html_dashboard(snapshot)

    html_ts     = DASH_DIR / f"hypersrl_dashboard_{RUN_TIMESTAMP}.html"
    html_latest = DASH_DIR / "hypersrl_dashboard_latest.html"
    html_ts.write_text(html_content, encoding="utf-8")
    html_latest.write_text(html_content, encoding="utf-8")
    logger.info("HTML saved: %s (%d KB)",
                html_ts.name, len(html_content)//1024)

    # --- PDF Report ---
    logger.info("--- BUILDING PDF REPORT ---")
    pdf_ts     = DASH_DIR / f"hypersrl_report_{RUN_TIMESTAMP}.pdf"
    pdf_latest = DASH_DIR / "hypersrl_report_latest.pdf"
    build_pdf_report(snapshot, pdf_ts)
    import shutil
    shutil.copy(pdf_ts, pdf_latest)
    logger.info("PDF saved: %s", pdf_ts.name)

    # --- Colab inline display ---
    try:
        from IPython.display import display, HTML, FileLink
        display(HTML(f"""
        <div style="font-family:Arial;padding:12px;background:#f8f9fa;
                    border-radius:8px;border-left:4px solid #3498db;">
          <b>Cell 38 Complete</b><br>
          <a href="{html_latest}" target="_blank">
            Open HTML Dashboard</a> &nbsp;|&nbsp;
          System risk: <b style="color:{
              RISK_COLORS.get(
                  snapshot['risk_report'].get('system_risk_level','UNKNOWN'),
                  '#95a5a6')
          };">{snapshot['risk_report'].get('system_risk_level','UNKNOWN')}</b>
        </div>
        """))
        display(FileLink(str(pdf_latest), result_html_prefix="Download PDF: "))
    except Exception:
        pass

    summary = {
        "timestamp" : RUN_TIMESTAMP,
        "outputs"   : {
            "html_latest"      : str(html_latest),
            "html_timestamped" : str(html_ts),
            "pdf_latest"       : str(pdf_latest),
            "pdf_timestamped"  : str(pdf_ts),
            "dashboard_log"    : str(DASH_LOG),
        },
        "panels"    : 7,
        "html_kb"   : len(html_content) // 1024,
        "pdf_path"  : str(pdf_latest),
    }
    logger.info("=" * 70)
    logger.info("CELL 38 COMPLETE")
    for k, v in summary["outputs"].items():
        logger.info("  %-20s : %s", k, v)
    logger.info("=" * 70)
    return summary


# =============================================================================
# CELL 38 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_38() -> None:
    """
    T01 — Snapshot loads all required keys
    T02 — Country series computed for all countries
    T03 — Cumulative series starts near zero at baseline
    T04 — P1 figure returns Plotly Figure object
    T05 — P2 gauge figure has 3 indicators
    T06 — P3 bar chart has correct number of bars
    T07 — P4 heatmap z-matrix shape correct
    T08 — P4 all compound probabilities in [0, 100]
    T09 — P5 acceleration bars match countries
    T10 — P6 table has all countries represented
    T11 — P7 HTML contains all three country names
    T12 — HTML dashboard contains all panel labels
    T13 — HTML dashboard is self-contained (no external CSS links)
    T14 — PDF file created and non-zero size
    T15 — Monitoring triggers section in recommendations
    T16 — Dashboard log written with correct columns
    T17 — Risk color mapping covers all levels
    T18 — Threshold projection: Bangladesh amber in ~7 years at 108mm rise
    T19 — HTML size reasonable (>100KB, <20MB)
    T20 — End-to-end pipeline on Drive data
    """
    PASS = "PASS"
    FAIL = "FAIL"
    results = []

    def record(tid, name, status, detail=""):
        results.append((tid, name, status, detail))
        marker = "[PASS]" if status == PASS else "[FAIL]"
        print(f"  {marker} {tid}: {name}" +
              (f" — {detail}" if detail else ""))

    print("\n" + "=" * 70)
    print("CELL 38 — TEST SUITE")
    print("=" * 70 + "\n")

    import plotly.graph_objects as go

    # T01
    test_id = "T01"
    try:
        snap = load_snapshot()
        for k in ["timestamp","baseline_date","risk_report",
                  "recommendations","md_text","country_series"]:
            assert k in snap, f"Missing: {k}"
        record(test_id, "Snapshot loads all required keys", PASS,
               f"keys={list(snap.keys())}")
    except Exception as e:
        record(test_id, "Snapshot loads all required keys", FAIL, str(e))
        snap = None

    if snap is None:
        print("  Aborting remaining tests — snapshot failed to load.")
        return

    # T02
    test_id = "T02"
    try:
        for country in COUNTRIES:
            assert country in snap["country_series"], \
                f"{country} missing from country_series"
            cs = snap["country_series"][country]
            assert len(cs["cumulative"]) > 0, f"{country}: empty cumulative"
        record(test_id, "Country series for all countries", PASS,
               f"countries={list(snap['country_series'].keys())}")
    except Exception as e:
        record(test_id, "Country series for all countries", FAIL, str(e))

    # T03
    test_id = "T03"
    try:
        for country in COUNTRIES:
            cs  = snap["country_series"].get(country)
            if cs is None: continue
            cum = np.array(cs["cumulative"])
            # The cumulative series near baseline should be close to 0
            assert cum.min() < 50, \
                f"{country}: cumulative never near 0 (min={cum.min():.1f})"
        record(test_id, "Cumulative series anchored near baseline", PASS)
    except Exception as e:
        record(test_id, "Cumulative series anchored near baseline", FAIL, str(e))

    # T04
    test_id = "T04"
    try:
        fig = _p1_slr_timeseries(snap)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        record(test_id, "P1 SLR time series figure", PASS,
               f"traces={len(fig.data)}")
    except Exception as e:
        record(test_id, "P1 SLR time series figure", FAIL, str(e))

    # T05
    test_id = "T05"
    try:
        fig = _p2_risk_gauges(snap)
        assert isinstance(fig, go.Figure)
        indicators = [t for t in fig.data if isinstance(t, go.Indicator)]
        assert len(indicators) == 3, f"Expected 3 indicators, got {len(indicators)}"
        record(test_id, "P2 gauge figure has 3 indicators", PASS)
    except Exception as e:
        record(test_id, "P2 gauge figure has 3 indicators", FAIL, str(e))

    # T06
    test_id = "T06"
    try:
        fig  = _p3_threshold_countdown(snap)
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        # 3 countries x 3 levels = 9 bars
        assert len(bars) == 9, f"Expected 9 bars, got {len(bars)}"
        record(test_id, "P3 countdown bars correct count", PASS,
               f"bars={len(bars)}")
    except Exception as e:
        record(test_id, "P3 countdown bars correct count", FAIL, str(e))

    # T07
    test_id = "T07"
    try:
        fig = _p4_compound_heatmap(snap)
        hm  = [t for t in fig.data if hasattr(t, "z")][0]
        z   = np.array(hm.z)
        assert z.shape == (len(COUNTRIES), len(PROJ_YEARS)), \
            f"Heatmap shape {z.shape} != ({len(COUNTRIES)},{len(PROJ_YEARS)})"
        record(test_id, "P4 heatmap shape correct", PASS,
               f"shape={z.shape}")
    except Exception as e:
        record(test_id, "P4 heatmap shape correct", FAIL, str(e))

    # T08
    test_id = "T08"
    try:
        fig = _p4_compound_heatmap(snap)
        hm  = [t for t in fig.data if hasattr(t, "z")][0]
        z   = np.array(hm.z)
        assert z.min() >= 0,   f"Negative probability: {z.min()}"
        assert z.max() <= 100, f"Probability > 100%: {z.max()}"
        record(test_id, "P4 compound probabilities in [0,100]", PASS,
               f"min={z.min():.2f}%, max={z.max():.2f}%")
    except Exception as e:
        record(test_id, "P4 compound probabilities in [0,100]", FAIL, str(e))

    # T09
    test_id = "T09"
    try:
        fig  = _p5_acceleration(snap)
        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bars) == 1
        assert len(bars[0].x) == len(COUNTRIES)
        record(test_id, "P5 acceleration bars match countries", PASS)
    except Exception as e:
        record(test_id, "P5 acceleration bars match countries", FAIL, str(e))

    # T10
    test_id = "T10"
    try:
        fig  = _p6_measures_table(snap)
        tbl  = [t for t in fig.data if isinstance(t, go.Table)][0]
        cell_vals = tbl.cells.values[0]   # First column = country labels
        for country in COUNTRIES:
            assert any(country in str(v) for v in cell_vals), \
                f"{country} missing from measures table"
        record(test_id, "P6 table has all countries", PASS)
    except Exception as e:
        record(test_id, "P6 table has all countries", FAIL, str(e))

    # T11
    test_id = "T11"
    try:
        p7 = _p7_recommendations_html(snap)
        for country in COUNTRIES:
            assert country in p7, f"{country} missing from P7 HTML"
        assert len(p7) > 500, "P7 HTML suspiciously short"
        record(test_id, "P7 HTML contains all country names", PASS,
               f"len={len(p7)}")
    except Exception as e:
        record(test_id, "P7 HTML contains all country names", FAIL, str(e))

    # T12
    test_id = "T12"
    try:
        html = build_html_dashboard(snap)
        for label in ["Panel 1", "Panel 2", "Panel 3",
                      "Panel 4", "Panel 5", "Panel 6", "Panel 7"]:
            assert label in html, f"Missing panel label: {label}"
        record(test_id, "HTML dashboard has all 7 panel labels", PASS)
    except Exception as e:
        record(test_id, "HTML dashboard has all 7 panel labels", FAIL, str(e))

    # T13
    test_id = "T13"
    try:
        html = build_html_dashboard(snap)
        # No external stylesheet links (self-contained)
        import re
        ext_css = re.findall(r'<link[^>]+stylesheet[^>]+href=["\']http', html)
        assert len(ext_css) == 0, \
            f"External CSS links found: {ext_css}"
        assert "<style>" in html, "Inline style block missing"
        record(test_id, "HTML dashboard is self-contained", PASS)
    except Exception as e:
        record(test_id, "HTML dashboard is self-contained", FAIL, str(e))

    # T14 — PDF creation
    test_id = "T14"
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        build_pdf_report(snap, tmp_path)
        assert tmp_path.exists(), "PDF file not created"
        pdf_size = tmp_path.stat().st_size
        assert pdf_size > 10_000, \
            f"PDF suspiciously small: {pdf_size} bytes"
        tmp_path.unlink()
        record(test_id, "PDF file created and non-zero size", PASS,
               f"size={pdf_size // 1024} KB")
    except Exception as e:
        record(test_id, "PDF file created and non-zero size", FAIL, str(e))

    # T15
    test_id = "T15"
    try:
        rec   = snap["recommendations"]
        trigs = rec.get("monitoring_triggers", [])
        assert len(trigs) >= 3, f"Expected >= 3 triggers, got {len(trigs)}"
        for trig in trigs:
            assert "trigger"         in trig
            assert "threshold"       in trig
            assert "response_action" in trig
        record(test_id, "Monitoring triggers in recommendations", PASS,
               f"count={len(trigs)}")
    except Exception as e:
        record(test_id, "Monitoring triggers in recommendations", FAIL, str(e))

    # T16
    test_id = "T16"
    try:
        log_dash_event("TEST", "PASS", "unit test row")
        assert DASH_LOG.exists()
        df_log = pd.read_csv(DASH_LOG)
        for col in ["timestamp","stage","status","detail","error"]:
            assert col in df_log.columns, f"Missing column: {col}"
        record(test_id, "Dashboard log written correctly", PASS,
               f"entries={len(df_log)}")
    except Exception as e:
        record(test_id, "Dashboard log written correctly", FAIL, str(e))

    # T17
    test_id = "T17"
    try:
        for level in ["GREEN","AMBER","RED","CRITICAL","UNKNOWN"]:
            assert level in RISK_COLORS, f"Missing color for: {level}"
            assert RISK_COLORS[level].startswith("#"), \
                f"Invalid color format: {RISK_COLORS[level]}"
        record(test_id, "Risk color mapping complete", PASS)
    except Exception as e:
        record(test_id, "Risk color mapping complete", FAIL, str(e))

    # T18
    test_id = "T18"
    try:
        # Bangladesh: rise=108.68, rate=5.6, amber=150
        # Expected: (150-108.68)/5.6 = 7.4 years
        rise     = 108.68
        rate     = SSP245_RATE_MM_YR["Bangladesh"]
        amber    = THRESHOLDS["Bangladesh"]["amber"]
        expected = round((amber - rise) / rate, 1)
        assert abs(expected - 7.4) < 0.2, \
            f"Projection mismatch: expected ~7.4yr, got {expected}yr"
        record(test_id, "Threshold projection — Bangladesh amber ~7.4yr", PASS,
               f"expected={expected}yr")
    except Exception as e:
        record(test_id, "Threshold projection — Bangladesh amber ~7.4yr",
               FAIL, str(e))

    # T19
    test_id = "T19"
    try:
        html    = build_html_dashboard(snap)
        html_kb = len(html) // 1024
        assert html_kb > 100,   f"HTML too small: {html_kb} KB"
        assert html_kb < 20000, f"HTML too large: {html_kb} KB"
        record(test_id, "HTML size reasonable", PASS,
               f"{html_kb} KB")
    except Exception as e:
        record(test_id, "HTML size reasonable", FAIL, str(e))

    # T20 — end-to-end
    test_id = "T20"
    try:
        summary = run_cell_38()
        assert Path(summary["outputs"]["html_latest"]).exists()
        assert Path(summary["outputs"]["pdf_latest"]).exists()
        assert summary["panels"] == 7
        assert summary["html_kb"] > 100
        record(test_id, "End-to-end pipeline on Drive data", PASS,
               f"html={summary['html_kb']}KB, "
               f"pdf={Path(summary['outputs']['pdf_latest']).stat().st_size//1024}KB")
    except Exception as e:
        record(test_id, "End-to-end pipeline on Drive data", FAIL, str(e))

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

test_cell_38()

print("FINAL DASHBOARD SUMMARY")
print("-" * 70)
dash_latest = DASH_DIR / "hypersrl_dashboard_latest.html"
pdf_latest  = DASH_DIR / "hypersrl_report_latest.pdf"
if dash_latest.exists():
    print(f"  HTML dashboard : {dash_latest}")
    print(f"  HTML size      : {dash_latest.stat().st_size // 1024} KB")
if pdf_latest.exists():
    print(f"  PDF report     : {pdf_latest}")
    print(f"  PDF size       : {pdf_latest.stat().st_size // 1024} KB")
print(f"  Dashboard log  : {DASH_LOG}")
