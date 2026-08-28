
from __future__ import annotations

import ast
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.getenv("ALPR_OUTPUT_ROOT", APP_ROOT.parent / "output")).resolve()
RUN_ROOT_ENV = os.getenv("ALPR_RUN_ROOT", "").strip()
REFRESH_SECONDS = max(1, int(os.getenv("REFRESH_SECONDS", "2")))
AUTO_DISCOVER = os.getenv("AUTO_DISCOVER_RUNS", "true").lower() == "true"

KNOWN_FILES = {
    "raw": "results_raw_detections.csv",
    "rich": "results_rich.csv",
    "interpolated": "results_rich_interpolated.csv",
    "preprocessing": "results_preprocessing_comparison.csv",
    "final": "Final_outputs.csv",
    "final_md": "Final_outputs.md",
    "metrics": "pipeline_metrics.md",
    "db": "results.db",
    "video": "annotated_output.mp4",
}
IMAGE_DIRS = ("plate_crops", "plate_crops_high_confidence")

st.set_page_config(
    page_title="ALPR Research Console",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;600;800&display=swap');

:root {
  --bg: #07090e; --panel: rgba(16, 24, 33, 0.65); --panel-hover: rgba(22, 33, 46, 0.85); --border: rgba(38, 52, 67, 0.5);
  --text: #eef4f8; --muted: #91a0ae; --blue: #00d2ff; --green: #00e676;
  --amber: #ffca28; --red: #ff5252;
}
* { font-family: 'Inter', sans-serif; }
h1, h2, h3, .metric-value, .big-plate, .stage-title { font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; }
[data-testid="stAppViewContainer"] { 
    background: radial-gradient(circle at top right, #111a28, #07090e 60%); 
}
[data-testid="stSidebar"] { 
    background: rgba(10, 15, 22, 0.8) !important; 
    backdrop-filter: blur(12px); 
    border-right: 1px solid var(--border); 
}
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 4rem; }
.small { color: var(--muted); font-size: 0.85rem; }
.eyebrow { 
    color: var(--blue); font-size: 0.75rem; font-weight: 800; letter-spacing: 0.2em; text-transform: uppercase;
    text-shadow: 0 0 10px rgba(0, 210, 255, 0.4);
}
.card {
  background: var(--panel);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--border); border-radius: 16px; padding: 20px;
  min-height: 110px; height: 100%; box-sizing: border-box;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
  transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px); border-color: rgba(0, 210, 255, 0.4);
    box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.15);
}
.metric-label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;}
.metric-value { 
    color: var(--text); font-size: 2rem; font-weight: 800; margin-top: 6px;
    background: linear-gradient(90deg, #fff, #b3d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-sub { color: var(--muted); font-size: 0.8rem; margin-top: 4px; }
.pill {
  display: inline-flex; align-items: center; justify-content: center; padding: 6px 12px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 800; border: 1px solid var(--border);
  background: rgba(21, 33, 44, 0.8); color: #cbd7e1;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2); backdrop-filter: blur(5px);
}
.pill-green { background: rgba(0, 230, 118, 0.1); color: #00e676; border-color: rgba(0, 230, 118, 0.3); box-shadow: 0 0 10px rgba(0, 230, 118, 0.2); }
.pill-amber { background: rgba(255, 202, 40, 0.1); color: #ffca28; border-color: rgba(255, 202, 40, 0.3); box-shadow: 0 0 10px rgba(255, 202, 40, 0.2); }
.pill-blue { background: rgba(0, 210, 255, 0.1); color: #00d2ff; border-color: rgba(0, 210, 255, 0.3); box-shadow: 0 0 10px rgba(0, 210, 255, 0.2); }
.pill-red { background: rgba(255, 82, 82, 0.1); color: #ff5252; border-color: rgba(255, 82, 82, 0.3); box-shadow: 0 0 10px rgba(255, 82, 82, 0.2); }
.stage {
  border: 1px solid var(--border); border-radius: 14px; background: var(--panel);
  padding: 16px; min-height: 95px; height: 100%; box-sizing: border-box;
  transition: all 0.3s ease; backdrop-filter: blur(10px);
}
.stage:hover { background: var(--panel-hover); border-color: rgba(255,255,255,0.2); }
.stage-title { font-weight: 800; font-size: 0.85rem; color: #fff; }
.stage-meta { color: var(--muted); font-size: 0.75rem; margin-top: 6px; line-height: 1.4; }
.callout {
  border-left: 4px solid var(--blue); background: linear-gradient(90deg, rgba(0, 210, 255, 0.1), transparent);
  padding: 16px 20px; border-radius: 0 10px 10px 0; color: #d1e0ec; font-size: 0.9rem;
}
.evidence {
  background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
  padding: 20px; text-align: center; backdrop-filter: blur(10px);
  height: 100%; box-sizing: border-box;
}
.big-plate { 
    font-size: 1.8rem; font-weight: 800; letter-spacing: 0.1em; color: #fff; 
    margin: 10px 0; text-shadow: 0 0 15px rgba(255,255,255,0.2);
}
.trace {
  background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 0.8rem; color: #a1b0c0;
}
.progress-container { width: 100%; background-color: #1a242f; border-radius: 999px; overflow: hidden; height: 12px; margin-top: 15px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3); }
.progress-bar { height: 100%; background: linear-gradient(90deg, #00d2ff, #3a7bd5); transition: width 0.4s ease; box-shadow: 0 0 10px rgba(0,210,255,0.5); }
.status-flex { display: flex; justify-content: space-between; align-items: center; margin-top: 20px; }
hr { border-color: var(--border); margin: 2rem 0; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(0, 210, 255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# File/run discovery
# ---------------------------------------------------------------------------
def is_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return any((p / name).exists() for name in KNOWN_FILES.values())

def discover_runs() -> list[Path]:
    candidates = []
    if RUN_ROOT_ENV:
        p = Path(RUN_ROOT_ENV).expanduser()
        if is_run_dir(p):
            candidates.append(p)
    root = OUTPUT_ROOT.expanduser()
    if root.exists() and root.is_dir() and AUTO_DISCOVER:
        if is_run_dir(root):
            candidates.append(root)
        else:
            for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
                if is_run_dir(p):
                    candidates.append(p)
    # bundled sample
    sample = APP_ROOT / "sample_run"
    if is_run_dir(sample):
        candidates.append(sample)
    # de-duplicate
    out, seen = [], set()
    for p in candidates:
        key = str(p.resolve()).lower()
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out

def run_label(p: Path) -> str:
    return p.name

def artifact_path(run: Path, key: str) -> Path:
    return run / KNOWN_FILES[key]

def artifact_exists(run: Path, key: str) -> bool:
    return artifact_path(run, key).exists()

def run_state(run: Path) -> tuple[str, str]:
    present = [k for k in KNOWN_FILES if artifact_exists(run, k)]
    if artifact_exists(run, "final") and artifact_exists(run, "db"):
        return "COMPLETE", "Core final artifacts are present."
    if present:
        return "IN PROGRESS / PARTIAL", "State inferred from currently present artifacts; the pipeline itself was not modified."
    return "NO ARTIFACTS", "No recognized pipeline artifacts found."

def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.2f}"

def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Artifact loaders
# Cache keys include modification time so live runs refresh naturally.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype("string")
    return df

@st.cache_data(show_spinner=False)
def read_db(path_str: str, mtime: float) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    con = sqlite3.connect(p)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            con,
        )["name"].tolist()
        if "ocr_results" not in tables:
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM ocr_results", con)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype("string")
        return df
    finally:
        con.close()

def load_artifacts(run: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for key in ("raw", "rich", "interpolated", "preprocessing", "final"):
        p = artifact_path(run, key)
        out[key] = read_csv(str(p), p.stat().st_mtime_ns if p.exists() else -1) if p.exists() else pd.DataFrame()
    p = artifact_path(run, "db")
    out["db"] = read_db(str(p), p.stat().st_mtime_ns if p.exists() else -1) if p.exists() else pd.DataFrame()
    return out

def read_metrics_text(run: Path) -> str:
    p = artifact_path(run, "metrics")
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def parse_bbox(value):
    if pd.isna(value):
        return None
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return None

def basename_from_path(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return Path(str(value).replace("\\", "/")).name

def find_artifact_image(run: Path, stored_path) -> Optional[Path]:
    name = basename_from_path(stored_path)
    if not name:
        return None
    # First try the original path exactly.
    original = Path(str(stored_path))
    if original.exists():
        return original
    # Then resolve by basename inside the portable run root.
    for dirname in IMAGE_DIRS:
        d = run / dirname
        if d.exists():
            direct = d / name
            if direct.exists():
                return direct
    # Last resort: recursive lookup by basename.
    for dirname in IMAGE_DIRS:
        d = run / dirname
        if d.exists():
            matches = list(d.rglob(name))
            if matches:
                return matches[0]
    return None

def artifact_inventory(run: Path) -> pd.DataFrame:
    rows = []
    for key, filename in KNOWN_FILES.items():
        p = run / filename
        rows.append({
            "artifact": filename,
            "purpose": key,
            "status": "AVAILABLE" if p.exists() else "MISSING",
            "size": f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "—",
            "modified": pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S") if p.exists() else "—",
        })
    for d in IMAGE_DIRS:
        p = run / d
        count = sum(1 for _ in p.rglob("*") if _.is_file()) if p.exists() else 0
        rows.append({"artifact": d + "/", "purpose": "images", "status": "AVAILABLE" if p.exists() else "MISSING", "size": f"{count} files", "modified": "—"})
    return pd.DataFrame(rows)

def read_status(run: Path) -> dict:
    p = run / "status.json"
    if p.exists():
        try:
            import json
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------
runs = discover_runs()
if not runs:
    st.error("No run directory found. Set ALPR_OUTPUT_ROOT or ALPR_RUN_ROOT in .env.")
    st.stop()

with st.sidebar:
    st.markdown("## ALPR Research Console")
    st.caption("Artifact-driven • pipeline remains unchanged")
    st.divider()
    run_options = {f"{run_label(p)}  ·  {str(p)}": p for p in runs}
    default_index = 0
    selected_label = st.selectbox("Run", list(run_options.keys()), index=default_index)
    run = run_options[selected_label]

    state, state_note = run_state(run)
    state_class = "pill-green" if state == "COMPLETE" else "pill-amber" if "PARTIAL" in state else "pill-red"
    st.markdown(f'<span class="pill {state_class}">● {state}</span>', unsafe_allow_html=True)
    st.caption(state_note)
    st.divider()

    page = st.radio(
        "Explore",
        ["Overview", "Pipeline", "Vehicles", "Run Artifacts"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Run root: `{run}`")
    st.caption(f"Refresh interval: {REFRESH_SECONDS}s")

# Optional live refresh. This only re-reads artifacts; it does not interact with the pipeline.
if st_autorefresh is not None and REFRESH_SECONDS > 0:
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="alpr_live_refresh")

data = load_artifacts(run)
raw = data["raw"]
rich = data["rich"]
interp = data["interpolated"]
prep = data["preprocessing"]
final = data["final"]
db = data["db"]

# ---------------------------------------------------------------------------
# Derived metrics: only mathematical transformations of persisted artifacts.
# ---------------------------------------------------------------------------
def success_df() -> pd.DataFrame:
    return db[db["status"].astype(str).str.upper() == "SUCCESS"].copy() if "status" in db else pd.DataFrame()

def funnel() -> dict:
    jobs = len(db)
    success = len(success_df())
    final_n = len(final)
    return {"jobs": jobs, "success": success, "final": final_n}

def unique_count(df, col):
    return int(df[col].nunique()) if not df.empty and col in df.columns else 0

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="eyebrow">RUN OVERVIEW</div>', unsafe_allow_html=True)
    st.title("ALPR Research Console")
    state, state_note = run_state(run)
    st.markdown(
        f'<span class="pill pill-blue">RUN {run.name}</span> '
        f'<span class="small">{state_note}</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    
    status_data = read_status(run)
    is_running = status_data.get("status") == "running"
    
    if is_running:
        pct = min(100, (status_data.get('frame_idx', 0) / max(status_data.get('total_frames', 1), 1)) * 100)
        st.markdown(f"""
        <div class="card" style="margin-bottom: 2rem; border-color: #00d2ff; animation: pulse 2s infinite;">
            <div class="eyebrow" style="margin-bottom: 5px;"><span class="pill pill-blue" style="margin-right: 10px;">LIVE</span> Pipeline is running</div>
            <div class="status-flex">
                <div>
                    <div style="font-family: 'Outfit'; font-size: 1.5rem; font-weight: 600; color: #fff;">Processing Frame {status_data.get('frame_idx', 0):,} / {status_data.get('total_frames', 0):,}</div>
                </div>
                <div style="text-align: right;">
                    <div class="small" style="color: #cbd7e1;">Active Vehicles: <b style="color: #fff;">{status_data.get('active_count', 0)}</b> | Finalized: <b style="color: #fff;">{status_data.get('finalized_count', 0)}</b></div>
                </div>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    jobs = len(db)
    successes = len(success_df())
    interp_rows = len(interp)
    raw_tracks = unique_count(raw, "car_id")

    st.write("")
    st.subheader("Recognition funnel")
    f = funnel()
    c = st.columns(5)
    steps = [
        ("RAW DETECTION RECORDS", len(raw), f"{raw_tracks} tracks"),
        ("RECOGNITION JOBS", f["jobs"], "SQLite ocr_results"),
        ("SUCCESS", f["success"], f"{(f['success']/f['jobs']*100):.1f}% of jobs" if f["jobs"] else "—"),
        ("FINAL OUTPUTS", f["final"], "Final_outputs.csv"),
        ("IMAGE EVIDENCE", sum(1 for d in IMAGE_DIRS if (run/d).exists()), "crop folders present"),
    ]
    for col, (lab, val, sub) in zip(c, steps):
        with col:
            metric_card(lab, fmt_num(val), sub)

    st.write("")
    st.subheader("Observable pipeline surface")
    stages = [
        ("01", "Frame ingestion", "video metadata / metrics file", "conditional"),
        ("02", "Detection + tracking", f"{fmt_num(raw_tracks)} track IDs represented", "available" if not raw.empty else "waiting"),
        ("2.5", "Plate association", f"{fmt_num(len(raw))} persisted raw records", "available" if not raw.empty else "waiting"),
        ("03", "Buffering", "active in-memory state is not persisted", "not persisted"),
        ("04", "Finalization", "finalize flags / rich tracks when present", "conditional"),
        ("05", "Job / frame selection", "downstream job evidence; candidate ranking is not fully persisted", "partial"),
        ("06", "OCR", f"{fmt_num(jobs)} SQLite recognition jobs", "available" if not db.empty else "waiting"),
        ("07", "Temporal fusion", f"{fmt_num(successes)} successful fusion records", "available" if not db.empty else "waiting"),
        ("08", "Database / analytics", "SQLite ocr_results", "available" if not db.empty else "waiting"),
        ("09", "Interpolation", f"{fmt_num(interp_rows)} rows", "available" if not interp.empty else "waiting"),
        ("10", "Visualization", "annotated_output.mp4 when supplied", "conditional"),
    ]
    for i in range(0, len(stages), 4):
        row = stages[i:i+4]
        cs = st.columns(len(row))
        for col, (num, name, detail, status) in zip(cs, row):
            dot = "●" if status == "available" else "◐" if status == "partial" or status == "conditional" else "○"
            with col:
                st.markdown(
                    f'<div class="stage"><div class="stage-title">{dot} {num} · {name}</div>'
                    f'<div class="stage-meta">{detail}<br><b>{status.upper()}</b></div></div>',
                    unsafe_allow_html=True,
                )

    st.write("")
    st.subheader("Run freshness")
    inv = artifact_inventory(run)
    st.dataframe(inv, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
elif page == "Pipeline":
    st.markdown('<div class="eyebrow">STAGE INVESTIGATOR</div>', unsafe_allow_html=True)
    st.title("Pipeline Explorer")
    stage_names = [
        "01 · Frame Ingestion", "02 · Detection & Tracking", "2.5 · Plate Association",
        "03 · Buffering", "04 · Finalization", "05 · Job / Frame Selection",
        "06 · OCR", "07 · Temporal Fusion", "08 · Database / Analytics",
        "09 · Interpolation", "10 · Visualization",
    ]
    stage = st.selectbox("Stage", stage_names)

    if stage.startswith("01"):
        st.subheader("Frame Ingestion")
        metrics = read_metrics_text(run)
        if metrics:
            st.code(metrics, language="markdown")
        else:
            st.info("pipeline_metrics.md is not present in this run. Ingestion runtime counters are therefore not available from the supplied artifacts.")
    elif stage.startswith("02"):
        st.subheader("Detection & Tracking — persisted view")
        c = st.columns(4)
        with c[0]: metric_card("Detection records", fmt_num(len(raw)))
        with c[1]: metric_card("Track IDs", fmt_num(unique_count(raw, "car_id")))
        with c[2]: metric_card("Frames represented", fmt_num(unique_count(raw, "frame_nmr")))
        with c[3]: metric_card("Records / track", f"{len(raw)/max(unique_count(raw,'car_id'),1):.1f}")
        if not raw.empty:
            per_track = raw.groupby("car_id").agg(
                first_frame=("frame_nmr","min"),
                last_frame=("frame_nmr","max"),
                observations=("frame_nmr","size"),
                mean_plate_det_conf=("license_plate_bbox_score","mean"),
            ).reset_index()
            per_track["span_frames"] = per_track["last_frame"] - per_track["first_frame"] + 1
            st.subheader("Track lifecycle summary")
            st.dataframe(per_track.sort_values("observations", ascending=False), width="stretch", hide_index=True)
    elif stage.startswith("2.5"):
        st.subheader("Plate Association — persisted records")
        c = st.columns(4)
        positive = int((raw["license_plate_bbox_score"] > 0).sum()) if "license_plate_bbox_score" in raw else 0
        with c[0]: metric_card("Persisted records", fmt_num(len(raw)))
        with c[1]: metric_card("Positive plate scores", fmt_num(positive))
        with c[2]: metric_card("Mean plate score", f"{raw['license_plate_bbox_score'].mean():.3f}" if not raw.empty else "—")
        with c[3]: metric_card("Tracks represented", fmt_num(unique_count(raw, "car_id")))
        st.dataframe(raw, width="stretch", hide_index=True)
    elif stage.startswith("03"):
        st.subheader("Active Buffering")
        st.warning("The existing pipeline does not persist active VehicleBuffer state. This console will not invent active-buffer counts. Finalized/observable track evidence is available under Vehicles.")
    elif stage.startswith("04"):
        st.subheader("Finalization — observable evidence")
        if not db.empty:
            cols = st.columns(4)
            with cols[0]: metric_card("OCR jobs", fmt_num(len(db)))
            with cols[1]: metric_card("Rows with finalize reason", fmt_num(db["finalize_reason"].notna().sum()) if "finalize_reason" in db else "—")
            with cols[2]: metric_card("Low diversity", fmt_num(pd.to_numeric(db["low_diversity"], errors="coerce").fillna(0).astype(bool).sum()) if "low_diversity" in db else "—")
            with cols[3]: metric_card("Possible ID switch", fmt_num(pd.to_numeric(db["possible_id_switch"], errors="coerce").fillna(0).astype(bool).sum()) if "possible_id_switch" in db else "—")
            cols_show = [c for c in ["track_id","status","finalize_reason","low_diversity","possible_id_switch","num_readings"] if c in db.columns]
            st.dataframe(db[cols_show], width="stretch", hide_index=True)
        else:
            st.info("SQLite artifact not present.")
    elif stage.startswith("05"):
        st.subheader("Job / Frame Selection")
        st.info("The current persisted artifacts expose downstream job results and number of readings, but not the complete candidate-frame ranking/selection trace. Those unavailable internals are intentionally not reconstructed.")
        if not prep.empty:
            c = st.columns(4)
            with c[0]: metric_card("Recognition jobs", fmt_num(len(prep)))
            with c[1]: metric_card("Jobs with readings", fmt_num((prep["num_readings"] > 0).sum()))
            with c[2]: metric_card("Mean readings / job", f"{prep['num_readings'].mean():.2f}")
            with c[3]: metric_card("Max readings / job", fmt_num(prep["num_readings"].max()))
            st.dataframe(prep.sort_values("num_readings", ascending=False), width="stretch", hide_index=True)
    elif stage.startswith("06"):
        st.subheader("OCR — worker output")
        if db.empty:
            st.info("results.db is not present.")
        else:
            c = st.columns(5)
            with c[0]: metric_card("Jobs", fmt_num(len(db)))
            with c[1]: metric_card("SUCCESS", fmt_num((db["status"] == "SUCCESS").sum()))
            with c[2]: metric_card("NO_PLATE", fmt_num((db["status"] == "NO_PLATE").sum()))
            with c[3]: metric_card("Mean successful confidence", f"{success_df()['confidence'].mean():.3f}" if len(success_df()) else "—")
            with c[4]: metric_card("Mean readings / success", f"{success_df()['num_readings'].mean():.2f}" if len(success_df()) else "—")
            show = [c for c in ["track_id","job_id","status","plate_text","confidence","num_readings","winner_branch"] if c in db.columns]
            st.dataframe(db[show], width="stretch", hide_index=True)
    elif stage.startswith("07"):
        st.subheader("Temporal Fusion")

        s = success_df()

        if s.empty:
            st.info("No SUCCESS records available.")

        else:
            # Fusion identity/confidence comes from SQLite.
            # Reading counts come from the preprocessing comparison artifact.
            reading_source = prep.copy()

            if "status" in reading_source.columns:
                reading_source = reading_source[
                    reading_source["status"].astype(str).str.upper() == "SUCCESS"
                ].copy()

            readings = pd.Series(dtype=float)

            if "num_readings" in reading_source.columns:
                readings = pd.to_numeric(
                    reading_source["num_readings"],
                    errors="coerce"
                ).dropna()

            c = st.columns(4)

            with c[0]:
                metric_card(
                    "Successful records",
                    fmt_num(len(s))
                )

            with c[1]:
                metric_card(
                    "Mean readings",
                    f"{readings.mean():.2f}" if not readings.empty else "—"
                )

            with c[2]:
                metric_card(
                    "Maximum readings",
                    fmt_num(readings.max()) if not readings.empty else "—"
                )

            with c[3]:
                metric_card(
                    "Mean fused confidence",
                    f"{s['confidence'].mean():.3f}"
                    if "confidence" in s.columns
                    else "—"
                )

            show = [
                c for c in [
                    "track_id",
                    "plate_text",
                    "confidence",
                    "plate_text_bilateral",
                    "conf_bilateral",
                    "plate_text_adaptive",
                    "conf_adaptive",
                    "winner_branch",
                    "stitched_track_ids"
                ]
                if c in s.columns
            ]

            st.dataframe(
                s.sort_values(
                    "confidence",
                    ascending=False
                )[show],
                width="stretch",
                hide_index=True
            )
    elif stage.startswith("08"):
        st.subheader("Database / Analytics")
        c = st.columns(4)
        with c[0]: metric_card("SQLite records", fmt_num(len(db)))
        with c[1]: metric_card("Distinct tracks", fmt_num(unique_count(db, "track_id")))
        with c[2]: metric_card("Distinct jobs", fmt_num(unique_count(db, "job_id")))
        with c[3]: metric_card("Valid records", fmt_num(int(pd.to_numeric(db["is_valid"], errors="coerce").fillna(0).sum())) if "is_valid" in db else "—")
        st.dataframe(db, width="stretch", hide_index=True)
    elif stage.startswith("09"):
        st.subheader("Interpolation")
        if interp.empty:
            st.info("results_rich_interpolated.csv is not present.")
        else:
            raw_rich = len(rich)
            ratio = len(interp) / raw_rich if raw_rich else np.nan
            c = st.columns(4)
            with c[0]: metric_card("Original rich rows", fmt_num(raw_rich))
            with c[1]: metric_card("Interpolated rows", fmt_num(len(interp)))
            with c[2]: metric_card("Row multiplier", f"{ratio:.2f}×" if np.isfinite(ratio) else "—")
            with c[3]: metric_card("Tracks", fmt_num(unique_count(interp, "car_id")))
            st.dataframe(interp.head(1000), width="stretch", hide_index=True)
    else:
        st.subheader("Visualization Output")
        video = artifact_path(run, "video")
        if video.exists():
            st.video(str(video))
        else:
            st.info("annotated_output.mp4 is not present in this run. The viewer will activate automatically when the artifact is supplied.")

# ---------------------------------------------------------------------------
# Vehicles
# ---------------------------------------------------------------------------
elif page == "Vehicles":
    st.markdown('<div class="eyebrow">TRACK INVESTIGATOR</div>', unsafe_allow_html=True)
    st.title("Vehicle Explorer")

    raw_ids = set(raw["car_id"].dropna().astype(int)) if "car_id" in raw else set()
    db_ids = set(pd.to_numeric(db["track_id"], errors="coerce").dropna().astype(int)) if "track_id" in db else set()
    final_ids = set(pd.to_numeric(final["Track_ID"], errors="coerce").dropna().astype(int)) if "Track_ID" in final else set()
    
    # Restrict to only high-confidence final outputs per user request
    ids = sorted(final_ids)

    if not ids:
        st.warning("No high-confidence track IDs are available in Final_outputs.csv.")
    else:
        selected = st.selectbox("Track ID", ids)
        raw_t = raw[raw["car_id"] == selected].copy() if "car_id" in raw else pd.DataFrame()
        rich_t = rich[rich["car_id"] == selected].copy() if "car_id" in rich else pd.DataFrame()
        db_t = db[pd.to_numeric(db["track_id"], errors="coerce") == selected].copy() if "track_id" in db else pd.DataFrame()
        final_t = final[pd.to_numeric(final["Track_ID"], errors="coerce") == selected].copy() if "Track_ID" in final else pd.DataFrame()

        plate = final_t["License_Plate"].iloc[0] if not final_t.empty and "License_Plate" in final_t else (
            db_t["plate_text"].iloc[0] if not db_t.empty and "plate_text" in db_t else "—"
        )
        conf = final_t["Confidence"].iloc[0] if not final_t.empty and "Confidence" in final_t else (
            db_t["confidence"].iloc[0] if not db_t.empty and "confidence" in db_t else np.nan
        )

        c = st.columns(5)
        with c[0]: metric_card("Track", str(selected), "cross-artifact identifier")
        with c[1]: metric_card("Raw observations", fmt_num(len(raw_t)))
        with c[2]: metric_card("Rich observations", fmt_num(len(rich_t)))
        with c[3]: metric_card("OCR jobs", fmt_num(len(db_t)))
        with c[4]: metric_card("Plate / confidence", f"{plate}", f"{float(conf):.4f}" if pd.notna(conf) else "—")

        if not raw_t.empty:
            first, last = raw_t["frame_nmr"].min(), raw_t["frame_nmr"].max()
            st.markdown(f"**Observed frame span:** `{first} → {last}`  ·  **span:** `{last-first+1} frames`")

        if not db_t.empty:
            st.subheader("Recognition chain")
            for _, r in db_t.iterrows():
                # Big Plate Visualizations for this job
                ev_cols = st.columns(3)
                for col, label, text_field, conf_field in [
                    (ev_cols[0], "BILATERAL", "plate_text_bilateral", "conf_bilateral"),
                    (ev_cols[1], "ADAPTIVE", "plate_text_adaptive", "conf_adaptive"),
                    (ev_cols[2], "FUSION", "plate_text", "confidence"),
                ]:
                    with col:
                        st.markdown(f'<div class="evidence"><b>{label}</b><div class="big-plate">{r.get(text_field) or "—"}</div><span class="small">confidence {float(r.get(conf_field,0)):.4f}</span></div>', unsafe_allow_html=True)
                
                st.write("")
                # Detailed Breakdown
                cols = st.columns(4)
                with cols[0]:
                    metric_card("Status", str(r.get("status", "—")), f"job {str(r.get('job_id','—'))[:8]}")
                with cols[1]:
                    metric_card("Fusion", str(r.get("plate_text") or "—"), f"confidence {float(r.get('confidence',0)):.4f}")
                with cols[2]:
                    metric_card("Bilateral", str(r.get("plate_text_bilateral") or "—"), f"{float(r.get('conf_bilateral',0)):.4f}")
                with cols[3]:
                    metric_card("Adaptive", str(r.get("plate_text_adaptive") or "—"), f"{float(r.get('conf_adaptive',0)):.4f}")
                st.markdown(
                    f'<div class="trace">winner={r.get("winner_branch","—")} · '
                    f'readings={r.get("num_readings","—")} · '
                    f'finalize_reason={r.get("finalize_reason","—")} · '
                    f'low_diversity={r.get("low_diversity","—")} · '
                    f'possible_id_switch={r.get("possible_id_switch","—")}</div>',
                    unsafe_allow_html=True,
                )
                st.divider()

        st.subheader("Recognition evidence files")
        if not final_t.empty:
            row = final_t.iloc[0]
            img_cols = st.columns(3)
            for col, label, field in [
                (img_cols[0], "Context", "Context_Image_Path"),
                (img_cols[1], "Bilateral", "Plate_Bilateral_Path"),
                (img_cols[2], "Adaptive", "Plate_Adaptive_Path"),
            ]:
                p = find_artifact_image(run, row.get(field))
                with col:
                    st.markdown(f"**{label}**")
                    if p and p.exists():
                        try:
                            with open(p, "rb") as f:
                                st.image(f.read(), width="stretch", caption=p.name)
                        except Exception:
                            st.warning("Could not read image file.")
                    else:
                        st.info("Image not found in this run directory.")

        if not rich_t.empty:
            st.subheader("Frame-level rich records")
            st.dataframe(rich_t, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Run artifacts
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="eyebrow">DATA CONTRACT</div>', unsafe_allow_html=True)
    st.title("Run Artifacts")
    st.caption("This page makes the provenance of every displayed number inspectable.")

    inv = artifact_inventory(run)
    st.dataframe(inv, width="stretch", hide_index=True)

    st.subheader("Artifact roots")
    st.code(str(run))

    tabs = st.tabs(["Final outputs", "Preprocessing", "Raw detections", "Rich", "Interpolated", "SQLite", "Metrics"])
    with tabs[0]:
        st.dataframe(final, width="stretch", hide_index=True)
    with tabs[1]:
        st.dataframe(prep, width="stretch", hide_index=True)
    with tabs[2]:
        st.dataframe(raw, width="stretch", hide_index=True)
    with tabs[3]:
        st.dataframe(rich, width="stretch", hide_index=True)
    with tabs[4]:
        st.dataframe(interp, width="stretch", hide_index=True)
    with tabs[5]:
        st.dataframe(db, width="stretch", hide_index=True)
    with tabs[6]:
        metrics = read_metrics_text(run)
        st.code(metrics if metrics else "pipeline_metrics.md not present.", language="markdown")
