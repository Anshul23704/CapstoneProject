from __future__ import annotations

import os
import sqlite3
import json
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

# =============================================================================
# Configuration
# =============================================================================
APP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.getenv("ALPR_OUTPUT_ROOT", APP_ROOT / "sample_run"))
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

# =============================================================================
# Styling
# =============================================================================
st.markdown(
    """
<style>
:root {
  --bg:#0a0f14; --panel:#101821; --border:#263443;
  --text:#eef4f8; --muted:#91a0ae; --blue:#67b7ff;
}
[data-testid="stAppViewContainer"] { background:var(--bg); }
[data-testid="stSidebar"] { background:#0d141c; border-right:1px solid var(--border); }
.block-container { max-width:1500px; padding-top:1.5rem; padding-bottom:4rem; }
h1,h2,h3 { letter-spacing:-.025em; }
.eyebrow { color:var(--blue); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.small { color:var(--muted); font-size:.82rem; }
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
.pill-red { background:#351820; color:#ff9aaa; border-color:#5b2732; }
.evidence { background:#0d151e; border:1px solid var(--border); border-radius:14px; padding:15px; }
.big-plate { font-size:1.55rem; font-weight:850; letter-spacing:.05em; }
.callout { border-left:3px solid var(--blue); background:#0e1923; padding:12px 15px; border-radius:7px; color:#c2ced8; }
.stage {
  border:1px solid var(--border); border-radius:13px; background:var(--panel);
  padding:13px 11px; min-height:72px;
}
.stage-title { font-weight:800; font-size:.80rem; }
.stage-meta { color:var(--muted); font-size:.68rem; margin-top:5px; line-height:1.35; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Run discovery / artifact access
# =============================================================================
def is_run_dir(p: Path) -> bool:
    return p.is_dir() and any((p / name).exists() for name in KNOWN_FILES.values())


def discover_runs() -> list[Path]:
    candidates: list[Path] = []

    if RUN_ROOT_ENV:
        p = Path(RUN_ROOT_ENV).expanduser()
        if is_run_dir(p):
            candidates.append(p)

    root = OUTPUT_ROOT.expanduser()
    if root.exists() and root.is_dir() and AUTO_DISCOVER:
        if is_run_dir(root):
            candidates.append(root)
        else:
            for p in sorted(
                root.iterdir(),
                key=lambda x: x.stat().st_mtime if x.exists() else 0,
                reverse=True,
            ):
                if is_run_dir(p):
                    candidates.append(p)

    # Development/sample data can sit beside the GUI.
    for sample_name in ("sample_run", "sample_runs"):
        sample = APP_ROOT / sample_name
        if is_run_dir(sample):
            candidates.append(sample)

    result: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()).lower()
        if key not in seen:
            result.append(p)
            seen.add(key)
    return result


def artifact_path(run: Path, key: str) -> Path:
    return run / KNOWN_FILES[key]


def run_state(run: Path) -> tuple[str, str]:
    present = [k for k in KNOWN_FILES if artifact_path(run, k).exists()]
    if artifact_path(run, "final").exists() and artifact_path(run, "db").exists():
        return "COMPLETE", "Core final artifacts are present."
    if present:
        return "PARTIAL", "State is inferred only from currently persisted artifacts."
    return "NO ARTIFACTS", "No recognized pipeline artifacts were found."


@st.cache_data(show_spinner=False)
def read_csv(path_str: str, mtime_ns: int) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_db(path_str: str, mtime_ns: int) -> pd.DataFrame:
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
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def load_artifacts(run: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for key in ("raw", "rich", "interpolated", "preprocessing", "final"):
        p = artifact_path(run, key)
        data[key] = read_csv(str(p), p.stat().st_mtime_ns) if p.exists() else pd.DataFrame()

    p = artifact_path(run, "db")
    data["db"] = read_db(str(p), p.stat().st_mtime_ns) if p.exists() else pd.DataFrame()
    return data


def basename_from_path(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return Path(str(value).replace("\\", "/")).name


def find_artifact_image(run: Path, stored_path) -> Optional[Path]:
    name = basename_from_path(stored_path)
    if not name:
        return None

    original = Path(str(stored_path))
    if original.exists():
        return original

    for dirname in IMAGE_DIRS:
        directory = run / dirname
        if not directory.exists():
            continue
        direct = directory / name
        if direct.exists():
            return direct
        matches = list(directory.rglob(name))
        if matches:
            return matches[0]
    return None


def fmt_num(value) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        value = float(value)
        return f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"
    except Exception:
        return str(value)


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def read_live_status(run: Path) -> dict | None:
    """
    Read live pipeline status written by main_pipeline.py.

    Returns None when this run does not have a status.json,
    so older/static runs continue to work normally.
    """
    status_path = run / "status.json"

    if not status_path.exists():
        return None

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None

        return data

    except (OSError, json.JSONDecodeError):
        return None

# =============================================================================
# Load selected run
# =============================================================================
runs = discover_runs()
if not runs:
    st.error(
        "No ALPR run was found. Set ALPR_OUTPUT_ROOT or ALPR_RUN_ROOT in .env, "
        "or place a sample run beside the GUI."
    )
    st.stop()

with st.sidebar:
    st.markdown("## ALPR Research Console")
    st.caption("Interactive view over persisted pipeline artifacts")
    st.divider()

    run_options = {f"{p.name} · {p}": p for p in runs}
    selected_label = st.selectbox("Pipeline run", list(run_options), index=0)
    run = run_options[selected_label]

    state, state_note = run_state(run)
    state_class = "pill-green" if state == "COMPLETE" else "pill-amber" if state == "PARTIAL" else "pill-red"
    st.markdown(f'<span class="pill {state_class}">● {state}</span>', unsafe_allow_html=True)
    st.caption(state_note)

    st.divider()
    page = st.radio(
        "Navigate",
        ["Dashboard", "Vehicles", "Recognition", "Analytics"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Run: `{run.name}`")
    st.caption(f"Refresh: {REFRESH_SECONDS}s")

# Refresh only causes artifact re-reading; it never modifies or controls the pipeline.
if st_autorefresh is not None and REFRESH_SECONDS > 0:
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="alpr_live_refresh")

data = load_artifacts(run)
raw = data["raw"]
rich = data["rich"]
interp = data["interpolated"]
prep = data["preprocessing"]
final = data["final"]
db = data["db"]

# =============================================================================
# Safe derived values
# =============================================================================
def success_df() -> pd.DataFrame:
    if db.empty or "status" not in db.columns:
        return pd.DataFrame()
    return db[db["status"].astype(str).str.upper() == "SUCCESS"].copy()


def unique_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].nunique()) if not df.empty and column in df.columns else 0


def final_track_ids() -> set[int]:
    if final.empty or "Track_ID" not in final.columns:
        return set()
    return set(pd.to_numeric(final["Track_ID"], errors="coerce").dropna().astype(int))


def all_track_ids() -> list[int]:
    ids: set[int] = set()
    if "car_id" in raw:
        ids |= set(pd.to_numeric(raw["car_id"], errors="coerce").dropna().astype(int))
    if "track_id" in db:
        ids |= set(pd.to_numeric(db["track_id"], errors="coerce").dropna().astype(int))
    ids |= final_track_ids()
    return sorted(ids)


def plate_detected_count() -> int:
    if "license_plate_bbox_score" in raw:
        scores = pd.to_numeric(raw["license_plate_bbox_score"], errors="coerce")
        return int((scores > 0).sum())
    return 0


def safe_confidence(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def get_track_df(df: pd.DataFrame, column: str, track_id: int) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame()
    ids = pd.to_numeric(df[column], errors="coerce")
    return df[ids == track_id].copy()

# =============================================================================
# DASHBOARD
# =============================================================================
if page == "Dashboard":
    st.markdown('<div class="eyebrow">RUN OVERVIEW</div>', unsafe_allow_html=True)
    st.title("ALPR Research Console")
    st.caption(f"Run: {run.name}")

    tracks = unique_count(raw, "car_id")
    jobs = len(db)
    successes = len(success_df())
    final_count = len(final)
    avg_conf = safe_confidence(success_df()["confidence"]) if not success_df().empty and "confidence" in success_df() else None

    cols = st.columns(4)
    with cols[0]:
        metric_card("Vehicles", fmt_num(tracks), "unique tracked vehicles")
    with cols[1]:
        metric_card("Plates detected", fmt_num(plate_detected_count()), "persisted plate detections")
    with cols[2]:
        metric_card("Successful OCR", fmt_num(successes), f"of {fmt_num(jobs)} recognition jobs")
    with cols[3]:
        metric_card("Avg. confidence", f"{avg_conf:.1%}" if avg_conf is not None else "—", "successful recognition records")

    st.write("")
    st.markdown(
        '<div class="callout"><b>Live artifact view:</b> this console reads the files and database produced by '
        'the existing ALPR pipeline. During a new run, persisted artifacts are re-read automatically as they change.</div>',
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # LIVE PIPELINE STATUS
    # ─────────────────────────────────────────────────────────────────────────────

    live_status = read_live_status(run)

    if live_status is not None:
        status = str(live_status.get("status", "")).lower()

        frame_idx = int(live_status.get("frame_idx", 0) or 0)
        total_frames = int(live_status.get("total_frames", 0) or 0)
        active_count = int(live_status.get("active_count", 0) or 0)
        finalized_count = int(live_status.get("finalized_count", 0) or 0)
        fused_success = int(live_status.get("fused_success", 0) or 0)
        no_plate = int(live_status.get("no_plate", 0) or 0)

        if total_frames > 0:
            progress = min(frame_idx / total_frames, 1.0)
        else:
            progress = 0.0

        if status == "running":
            st.markdown("### 🔴 Live Pipeline")

            st.caption("Pipeline is currently processing this run.")

            st.progress(
                progress,
                text=f"Processing frame {frame_idx:,} / {total_frames:,}"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Active Vehicles", f"{active_count:,}")

            with c2:
                st.metric("Finalized", f"{finalized_count:,}")

            with c3:
                st.metric("Successful OCR", f"{fused_success:,}")

            with c4:
                st.metric("No Plate", f"{no_plate:,}")

        else:
            st.markdown("### ✅ Pipeline Status")

            if total_frames > 0:
                st.progress(
                    progress,
                    text=f"Processed {frame_idx:,} / {total_frames:,} frames"
                )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Active Vehicles", f"{active_count:,}")

            with c2:
                st.metric("Finalized", f"{finalized_count:,}")

            with c3:
                st.metric("Successful OCR", f"{fused_success:,}")

            with c4:
                st.metric("No Plate", f"{no_plate:,}")

    else:
        # Older runs may not contain status.json.
        st.info(
            "Live pipeline status is unavailable for this run. "
            "Showing persisted pipeline results."
        )

    st.write("")
    st.subheader("Pipeline")
    stages = [
        ("01", "Ingestion", (not raw.empty) or artifact_path(run, "metrics").exists()),
        ("02", "Detection", not raw.empty),
        ("03", "Tracking", unique_count(raw, "car_id") > 0),
        ("04", "Plate", plate_detected_count() > 0),
        ("05", "OCR", not db.empty),
        ("06", "Fusion", len(success_df()) > 0),
        ("07", "Output", not final.empty),
    ]
    stage_cols = st.columns(len(stages))
    for col, (num, name, available) in zip(stage_cols, stages):
        with col:
            dot = "●" if available else "○"
            st.markdown(
                f'<div class="stage"><div class="stage-title">{dot} {num} · {name}</div>'
                f'<div class="stage-meta">{"artifact available" if available else "awaiting artifact"}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Final recognitions")
    if final.empty:
        st.info("No final outputs are currently available.")
    else:
        display_cols = [c for c in ["Track_ID", "License_Plate", "Confidence"] if c in final.columns]
        view = final[display_cols].copy()
        if "Confidence" in view.columns:
            view["Confidence"] = pd.to_numeric(view["Confidence"], errors="coerce").map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
        st.dataframe(view, use_container_width=True, hide_index=True)

# =============================================================================
# VEHICLES
# =============================================================================
elif page == "Vehicles":
    st.markdown('<div class="eyebrow">VEHICLE INVESTIGATION</div>', unsafe_allow_html=True)
    st.title("Vehicles")
    st.caption("Select a tracked vehicle and follow its persisted recognition evidence.")

    ids = all_track_ids()
    if not ids:
        st.info("No vehicle track IDs are available in this run yet.")
        st.stop()

    selected = st.selectbox("Vehicle", ids, format_func=lambda x: f"Vehicle / Track {x}")
    raw_t = get_track_df(raw, "car_id", selected)
    rich_t = get_track_df(rich, "car_id", selected)
    db_t = get_track_df(db, "track_id", selected)
    final_t = get_track_df(final, "Track_ID", selected)

    plate = "—"
    conf: Optional[float] = None
    if not final_t.empty:
        if "License_Plate" in final_t.columns:
            plate = final_t["License_Plate"].iloc[0]
        if "Confidence" in final_t.columns:
            vals = pd.to_numeric(final_t["Confidence"], errors="coerce").dropna()
            if not vals.empty:
                conf = float(vals.iloc[0])
    elif not db_t.empty:
        if "plate_text" in db_t.columns:
            plate = db_t["plate_text"].iloc[0]
        if "confidence" in db_t.columns:
            vals = pd.to_numeric(db_t["confidence"], errors="coerce").dropna()
            if not vals.empty:
                conf = float(vals.iloc[0])

    cols = st.columns(4)
    with cols[0]: metric_card("Track", str(selected), "vehicle identifier")
    with cols[1]: metric_card("Observations", fmt_num(len(raw_t)), "raw detection records")
    with cols[2]: metric_card("Recognition jobs", fmt_num(len(db_t)), "persisted OCR records")
    with cols[3]: metric_card("Confidence", f"{conf:.1%}" if conf is not None else "—", "final / fused result")

    st.write("")
    st.markdown("### Recognition result")
    result_cols = st.columns([2, 1])
    with result_cols[0]:
        st.markdown(
            f'<div class="evidence"><div class="small">RECOGNIZED PLATE</div>'
            f'<div class="big-plate">{plate if pd.notna(plate) else "—"}</div>'
            f'<div class="small">{f"Confidence: {conf:.1%}" if conf is not None else "No final confidence available"}</div></div>',
            unsafe_allow_html=True,
        )
    with result_cols[1]:
        status = "Recognized" if conf is not None or plate != "—" else "Unresolved"
        metric_card("Status", status, "based on persisted records")

    if not raw_t.empty and "frame_nmr" in raw_t.columns:
        frames = pd.to_numeric(raw_t["frame_nmr"], errors="coerce").dropna()
        if not frames.empty:
            st.caption(f"Observed frame span: {int(frames.min())} → {int(frames.max())} · {int(frames.max()-frames.min()+1)} frames")

    if not db_t.empty:
        st.markdown("### Recognition evidence")
        # Show the most useful persisted recognition fields without exposing the entire DB schema.
        latest = db_t.iloc[-1]
        evidence_cols = st.columns(3)
        for col, label, text_field, conf_field in [
            (evidence_cols[0], "Bilateral", "plate_text_bilateral", "conf_bilateral"),
            (evidence_cols[1], "Adaptive", "plate_text_adaptive", "conf_adaptive"),
            (evidence_cols[2], "Fusion", "plate_text", "confidence"),
        ]:
            text_value = latest.get(text_field, "—")
            confidence = pd.to_numeric(pd.Series([latest.get(conf_field)]), errors="coerce").iloc[0]
            with col:
                st.markdown(
                    f'<div class="evidence"><b>{label}</b><div class="big-plate">'
                    f'{text_value if pd.notna(text_value) else "—"}</div>'
                    f'<span class="small">confidence {confidence:.1%}</span></div>',
                    unsafe_allow_html=True,
                )

    # Actual image evidence from Final_outputs.csv when the paths are persisted.
    if not final_t.empty:
        row = final_t.iloc[0]
        image_fields = [
            ("Context", "Context_Image_Path"),
            ("Bilateral", "Plate_Bilateral_Path"),
            ("Adaptive", "Plate_Adaptive_Path"),
        ]
        available_images = [(label, find_artifact_image(run, row.get(field))) for label, field in image_fields]
        available_images = [(label, path) for label, path in available_images if path]

        if available_images:
            st.markdown("### Visual evidence")
            image_cols = st.columns(len(available_images))
            for col, (label, path) in zip(image_cols, available_images):
                with col:
                    st.markdown(f"**{label}**")
                    st.image(str(path), use_container_width=True)

    with st.expander("Technical details", expanded=False):
        st.caption("These details remain available for investigation without occupying the main interface.")
        if not rich_t.empty:
            st.write("Rich observations")
            st.dataframe(rich_t, use_container_width=True, hide_index=True)
        if not db_t.empty:
            st.write("Recognition records")
            st.dataframe(db_t, use_container_width=True, hide_index=True)

# =============================================================================
# RECOGNITION
# =============================================================================
elif page == "Recognition":
    st.markdown('<div class="eyebrow">RECOGNITION ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Recognition")
    st.caption("A compact view of how persisted recognition evidence behaves.")

    successes = success_df()
    jobs = len(db)
    success_n = len(successes)
    final_n = len(final)

    st.subheader("Recognition funnel")
    funnel_df = pd.DataFrame(
        {
            "count": [jobs, success_n, final_n],
        },
        index=["Recognition jobs", "Successful OCR", "Final outputs"],
    )
    st.bar_chart(funnel_df, y="count", height=260)

    c = st.columns(3)
    with c[0]: metric_card("Jobs", fmt_num(jobs), "persisted recognition records")
    with c[1]: metric_card("Successful", fmt_num(success_n), f"{success_n/jobs:.1%} of jobs" if jobs else "—")
    with c[2]: metric_card("Final outputs", fmt_num(final_n), "persisted final records")

    st.subheader("Preprocessing behaviour")
    if prep.empty or "status" not in prep.columns or "winner_branch" not in prep.columns:
        st.info("Preprocessing comparison data is not available in this run.")
    else:
        successful_p = prep[prep["status"].astype(str).str.upper() == "SUCCESS"].copy()
        winner_counts = successful_p["winner_branch"].astype(str).value_counts()
        if winner_counts.empty:
            st.info("No successful preprocessing comparison records are available.")
        else:
            st.bar_chart(winner_counts, height=240)
            st.caption("Winner labels are read from the persisted preprocessing comparison artifact; this is not an accuracy measurement.")

    st.subheader("Temporal evidence")
    reading_source = prep if not prep.empty else successes
    if not reading_source.empty and "num_readings" in reading_source.columns:
        temporal = reading_source.copy()
        if "status" in temporal.columns:
            temporal = temporal[temporal["status"].astype(str).str.upper() == "SUCCESS"]
        readings = pd.to_numeric(temporal["num_readings"], errors="coerce").dropna()
        if not readings.empty:
            c = st.columns(3)
            with c[0]: metric_card("Mean readings", f"{readings.mean():.2f}", "per successful record")
            with c[1]: metric_card("Median readings", f"{readings.median():.1f}", "per successful record")
            with c[2]: metric_card("Maximum readings", fmt_num(readings.max()), "per successful record")
            counts = readings.astype(int).value_counts().sort_index()
            st.bar_chart(counts, height=220)
        else:
            st.info("The artifact exists but contains no usable reading-count values.")
    else:
        st.info("Temporal reading counts are not persisted in the available recognition artifact for this run.")

    with st.expander("Recognition records", expanded=False):
        if successes.empty:
            st.info("No successful recognition records are available.")
        else:
            show = [
                c for c in [
                    "track_id", "plate_text", "confidence", "winner_branch", "num_readings",
                    "plate_text_bilateral", "conf_bilateral", "plate_text_adaptive", "conf_adaptive",
                ] if c in successes.columns
            ]
            st.dataframe(successes[show], use_container_width=True, hide_index=True)

# =============================================================================
# ANALYTICS
# =============================================================================
else:
    st.markdown('<div class="eyebrow">RESEARCH ANALYTICS</div>', unsafe_allow_html=True)
    st.title("Analytics")
    st.caption("Three descriptive views derived only from persisted pipeline artifacts.")

    successes = success_df()

    # 1. Recognition funnel
    st.subheader("1 · Recognition funnel")
    funnel_df = pd.DataFrame(
        {"count": [len(db), len(successes), len(final)]},
        index=["Recognition jobs", "Successful OCR", "Final outputs"],
    )
    st.bar_chart(funnel_df, y="count", height=260)

    # 2. Confidence distribution
    st.subheader("2 · Recognition confidence")
    if not successes.empty and "confidence" in successes.columns:
        confidence = pd.to_numeric(successes["confidence"], errors="coerce").dropna()
        if not confidence.empty:
            bins = np.linspace(0, 1, 11)
            counts, edges = np.histogram(confidence, bins=bins)
            labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
            chart = pd.DataFrame({"records": counts}, index=labels)
            st.bar_chart(chart, y="records", height=240)
        else:
            st.info("No usable confidence values are available.")
    else:
        st.info("Successful recognition confidence is not available in this run.")

    # 3. Preprocessing comparison
    st.subheader("3 · Preprocessing branch behaviour")
    if not prep.empty and "status" in prep.columns and "winner_branch" in prep.columns:
        successful_p = prep[prep["status"].astype(str).str.upper() == "SUCCESS"].copy()
        if not successful_p.empty:
            winner_counts = successful_p["winner_branch"].astype(str).value_counts()
            st.bar_chart(winner_counts, height=240)
            st.caption("Descriptive comparison of persisted winner labels. No ground-truth accuracy claim is made.")
        else:
            st.info("No successful preprocessing records are available.")
    else:
        st.info("Preprocessing comparison data is not available in this run.")

    st.markdown(
        '<div class="callout"><b>Research boundary:</b> accuracy, precision, recall, character-level accuracy, '
        'and improvement percentages are intentionally not shown without a ground-truth evaluation artifact. '
        'The displayed values are direct or deterministic summaries of persisted run data.</div>',
        unsafe_allow_html=True,
    )
