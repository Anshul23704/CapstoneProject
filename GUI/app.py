
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
OUTPUT_ROOT = Path(os.getenv("ALPR_OUTPUT_ROOT", APP_ROOT / "sample_runs"))
RUN_ROOT_ENV = os.getenv("ALPR_RUN_ROOT", "").strip()
REFRESH_SECONDS = max(1, int(os.getenv("REFRESH_SECONDS", "10")))
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
:root {
  --bg:#0a0f14; --panel:#101821; --panel2:#131e29; --border:#263443;
  --text:#eef4f8; --muted:#91a0ae; --blue:#67b7ff; --green:#5be08a;
  --amber:#f4c66d; --red:#ff7788;
}
[data-testid="stAppViewContainer"] { background:var(--bg); }
[data-testid="stSidebar"] { background:#0d141c; border-right:1px solid var(--border); }
.block-container { max-width:1500px; padding-top:1.4rem; padding-bottom:4rem; }
h1,h2,h3 { letter-spacing:-.025em; }
.small { color:var(--muted); font-size:.82rem; }
.eyebrow { color:var(--blue); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.card {
  background:linear-gradient(145deg,var(--panel),#0d151e);
  border:1px solid var(--border); border-radius:15px; padding:17px 19px;
  min-height:105px;
}
.metric-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; }
.metric-value { color:var(--text); font-size:1.72rem; font-weight:800; margin-top:4px; }
.metric-sub { color:var(--muted); font-size:.76rem; margin-top:3px; }
.pill {
  display:inline-block; padding:4px 9px; border-radius:999px;
  font-size:.72rem; font-weight:800; border:1px solid var(--border);
  background:#15212c; color:#cbd7e1;
}
.pill-green { background:#123022; color:#78e6a0; border-color:#1d5136; }
.pill-amber { background:#322712; color:#f2cb77; border-color:#5b451c; }
.pill-blue { background:#102b42; color:#91d0ff; border-color:#1c4e72; }
.pill-red { background:#351820; color:#ff9aaa; border-color:#5b2732; }
.stage {
  border:1px solid var(--border); border-radius:13px; background:var(--panel);
  padding:13px 11px; min-height:88px;
}
.stage-title { font-weight:800; font-size:.80rem; }
.stage-meta { color:var(--muted); font-size:.68rem; margin-top:5px; line-height:1.35; }
.stage-dot { font-size:.65rem; margin-right:5px; }
.callout {
  border-left:3px solid var(--blue); background:#0e1923; padding:12px 15px;
  border-radius:7px; color:#c2ced8;
}
.evidence {
  background:#0d151e; border:1px solid var(--border); border-radius:14px;
  padding:15px;
}
.big-plate { font-size:1.55rem; font-weight:850; letter-spacing:.05em; }
.trace {
  background:#0c131a; border:1px solid var(--border); border-radius:10px;
  padding:10px 12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.78rem;
}
hr { border-color:var(--border); }
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
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

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
        return pd.read_sql_query("SELECT * FROM ocr_results", con)
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
        ["Overview", "Pipeline", "Vehicles", "Evidence", "Analytics", "Run Artifacts"],
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

    total_raw = len(raw)
    raw_tracks = unique_count(raw, "car_id")
    frames = unique_count(raw, "frame_nmr")
    jobs = len(db)
    successes = len(success_df())
    final_n = len(final)
    rich_tracks = unique_count(rich, "car_id")
    interp_rows = len(interp)

    cols = st.columns(6)
    for c, label, value, sub in [
        (cols[0], "Raw detection records", fmt_num(total_raw), f"{raw_tracks} track IDs"),
        (cols[1], "Frames represented", fmt_num(frames), "raw detection artifact"),
        (cols[2], "Recognition jobs", fmt_num(jobs), f"{successes} SUCCESS"),
        (cols[3], "Successful jobs", fmt_num(successes), f"{jobs-successes} NO_PLATE"),
        (cols[4], "Final outputs", fmt_num(final_n), "Final_outputs.csv"),
        (cols[5], "Interpolated rows", fmt_num(interp_rows), f"{rich_tracks} rich tracks"),
    ]:
        with c:
            metric_card(label, value, sub)

    st.write("")
    st.markdown(
        '<div class="callout"><b>What this console is:</b> a run-agnostic investigation layer over the '
        'existing pipeline artifacts. It does not run or modify YOLO/OCR/tracking. Values are either '
        'read directly from persisted artifacts or explicitly derived from them.</div>',
        unsafe_allow_html=True,
    )

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
    ids = sorted(raw_ids | db_ids | final_ids)

    if not ids:
        st.warning("No track IDs are available.")
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
                cols = st.columns(3)
                with cols[0]:
                    metric_card("Status", str(r.get("status", "—")), f"job {str(r.get('job_id','—'))[:8]}")
                with cols[1]:
                    metric_card("Fusion", str(r.get("plate_text") or "—"), f"confidence {float(r.get('confidence',0)):.4f}")
                with cols[2]:
                    metric_card("Bilateral", str(r.get("plate_text_bilateral") or "—"), f"{float(r.get('conf_bilateral',0)):.4f}")
                st.markdown(
                    f'<div class="trace">winner={r.get("winner_branch","—")} · '
                    f'readings={r.get("num_readings","—")} · '
                    f'finalize_reason={r.get("finalize_reason","—")} · '
                    f'low_diversity={r.get("low_diversity","—")} · '
                    f'possible_id_switch={r.get("possible_id_switch","—")}</div>',
                    unsafe_allow_html=True,
                )

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
                    if p:
                        st.image(str(p), width="stretch", caption=p.name)
                    else:
                        st.info("Image not found in this run directory.")

        if not rich_t.empty:
            st.subheader("Frame-level rich records")
            st.dataframe(rich_t, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
elif page == "Evidence":
    st.markdown('<div class="eyebrow">EVIDENCE INSPECTOR</div>', unsafe_allow_html=True)
    st.title("Recognition Evidence")
    s = success_df()
    if s.empty:
        st.info("No SUCCESS recognition records are available.")
    else:
        options = s.apply(lambda r: f"Track {int(r.track_id)} · {r.plate_text} · {float(r.confidence):.4f}", axis=1).tolist()
        idx = st.selectbox("Recognition record", range(len(options)), format_func=lambda i: options[i])
        row = s.iloc[idx]

        c = st.columns(4)
        with c[0]: metric_card("Final plate", str(row.plate_text or "—"))
        with c[1]: metric_card("Fused confidence", f"{float(row.confidence):.4f}")
        with c[2]: metric_card("Winner", str(row.winner_branch))
        with c[3]: metric_card("Readings", fmt_num(row.num_readings))

        st.write("")
        cols = st.columns(2)
        for col, label, text_field, conf_field in [
            (cols[0], "BILATERAL", "plate_text_bilateral", "conf_bilateral"),
            (cols[1], "FUSION", "plate_text", "confidence"),
        ]:
            with col:
                st.markdown(f'<div class="evidence"><b>{label}</b><div class="big-plate">{row.get(text_field) or "—"}</div><span class="small">confidence {float(row.get(conf_field,0)):.4f}</span></div>', unsafe_allow_html=True)

        st.write("")
        final_t = final[pd.to_numeric(final["Track_ID"], errors="coerce") == int(row.track_id)] if not final.empty else pd.DataFrame()
        if not final_t.empty:
            r = final_t.iloc[0]
            st.subheader("Actual image evidence")
            cols = st.columns(2)
            for col, label, field in [
                (cols[0], "Context frame", "Context_Image_Path"),
                (cols[1], "Bilateral crop", "Plate_Bilateral_Path"),
            ]:
                p = find_artifact_image(run, r.get(field))
                with col:
                    st.markdown(f"**{label}**")
                    if p:
                        st.image(str(p), width="stretch", caption=p.name)
                    else:
                        st.warning("The CSV references this image, but the image file is not available under the selected run root.")
        else:
            st.info("Final_outputs.csv does not contain this track; therefore no image paths are available from that artifact.")

        st.subheader("Raw persisted recognition record")
        st.dataframe(row.to_frame("value"), width="stretch")

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
elif page == "Analytics":
    st.markdown('<div class="eyebrow">RESEARCH ANALYTICS</div>', unsafe_allow_html=True)
    st.title("Run Analytics")
    s = success_df()

    # Funnel
    st.subheader("1 · Recognition funnel")
    if not db.empty:
        funnel_df = pd.DataFrame({
            "stage": ["Recognition jobs", "SUCCESS", "Final outputs"],
            "count": [len(db), len(s), len(final)],
        }).set_index("stage")
        st.bar_chart(funnel_df, y="count", height=280)

    # Preprocessing
    st.subheader("2 · Preprocessing branch behaviour")
    if not prep.empty:
        successful_p = prep[prep["status"] == "SUCCESS"].copy()
        winner_counts = successful_p["winner_branch"].value_counts()
        cols = st.columns(2)
        with cols[0]: metric_card("Successful jobs", fmt_num(len(successful_p)))
        with cols[1]: metric_card("Bilateral wins", fmt_num(int((successful_p["winner_branch"]=="bilateral").sum())))
        st.bar_chart(winner_counts, height=260)
        st.caption("Winner labels are read from results_preprocessing_comparison.csv; no accuracy claim is implied.")

    # Temporal evidence
    st.subheader("3 · Temporal evidence")

    # Reading counts are persisted in the preprocessing-comparison artifact,
    # not necessarily in the SQLite schema. Do not assume the two schemas match.
    if not prep.empty and "num_readings" in prep.columns:

        temporal = prep.copy()

        if "status" in temporal.columns:
            temporal = temporal[
                temporal["status"].astype(str).str.upper() == "SUCCESS"
            ].copy()

        readings = pd.to_numeric(
            temporal["num_readings"],
            errors="coerce"
        ).dropna()

        if not readings.empty:

            cols = st.columns(4)

            with cols[0]:
                metric_card(
                    "Successful jobs",
                    fmt_num(len(readings))
                )

            with cols[1]:
                metric_card(
                    "Mean readings",
                    f"{readings.mean():.2f}"
                )

            with cols[2]:
                metric_card(
                    "Median readings",
                    f"{readings.median():.1f}"
                )

            with cols[3]:
                metric_card(
                    "Maximum readings",
                    fmt_num(readings.max())
                )

            counts = (
                readings
                .astype(int)
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                counts,
                height=250
            )

            st.caption(
                "Descriptive evidence from the persisted preprocessing "
                "comparison artifact. This does not by itself establish "
                "that additional readings improve recognition accuracy."
            )

        else:
            st.info(
                "The preprocessing artifact exists, but contains no "
                "usable reading-count values for SUCCESS records."
            )

    else:
        st.info(
            "Temporal reading counts are unavailable because "
            "results_preprocessing_comparison.csv does not expose "
            "a num_readings column in this run."
        )

        # Confidence comparison
        st.subheader("4 · Confidence comparison")
        if not prep.empty:
            comp = prep[prep["status"]=="SUCCESS"][["track_id","conf_bilateral","confidence"]].copy()
            comp = comp.set_index("track_id")
            st.line_chart(comp, height=300)

        # Trajectory expansion
        st.subheader("5 · Trajectory interpolation")
        if not rich.empty and not interp.empty:
            ratio = len(interp) / len(rich)
            c = st.columns(3)
            with c[0]: metric_card("Original rich rows", fmt_num(len(rich)))
            with c[1]: metric_card("Interpolated rows", fmt_num(len(interp)))
            with c[2]: metric_card("Row multiplier", f"{ratio:.2f}×")
            track_options = sorted(interp["car_id"].dropna().astype(int).unique()) if "car_id" in interp else []
            if track_options:
                tid = st.selectbox("Trajectory track", track_options, key="trajectory_track")
                rt = rich[rich["car_id"] == tid].copy()
                it = interp[interp["car_id"] == tid].copy()
                if not rt.empty:
                    st.caption(f"Track {tid}: raw observations vs interpolated trajectory.")
                    chart = pd.DataFrame({
                        "raw": pd.Series(rt["frame_nmr"].values),
                        "interpolated": pd.Series(it["frame_nmr"].head(len(rt)).values),
                    })
                    # Keep the chart simple and honest; the detailed records are below.
                    st.dataframe(pd.concat([
                        rt.assign(source="raw"),
                        it.assign(source="interpolated")
                    ], ignore_index=True).head(500), width="stretch", hide_index=True)

        st.subheader("6 · Failure / quality flags")
        if not db.empty:
            c = st.columns(3)
            low = int(pd.to_numeric(db["low_diversity"], errors="coerce").fillna(0).astype(bool).sum()) if "low_diversity" in db else 0
            switch = int(pd.to_numeric(db["possible_id_switch"], errors="coerce").fillna(0).astype(bool).sum()) if "possible_id_switch" in db else 0
            no_plate = int((db["status"]=="NO_PLATE").sum()) if "status" in db else 0
            with c[0]: metric_card("NO_PLATE jobs", fmt_num(no_plate))
            with c[1]: metric_card("Low-diversity flags", fmt_num(low))
            with c[2]: metric_card("Possible ID-switch flags", fmt_num(switch))

        st.markdown(
            '<div class="callout"><b>Research boundary:</b> this console deliberately does not display '
            'accuracy, precision, recall, character accuracy, or improvement percentages because the supplied '
            'run contains no ground-truth annotation artifact. Those belong in the later controlled evaluation layer.</div>',
            unsafe_allow_html=True,
        )

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
