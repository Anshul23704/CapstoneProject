"""
Stage 8 — Database Storage & Analytics.

FIX vs previous version
──────────────────────────
Schema is extended to store plate_bbox, the number of per-frame readings
that went into the fusion, low_diversity, and possible_id_switch. Stage 4
and Stage 6/7 already compute all of this; the old schema threw it away,
so the analytics report had no way to, e.g., break down accuracy by
possible ID-switch tracks. `is_valid` is still stored as given by the
caller — main_pipeline.py now passes the actual Stage 7 fusion validity
instead of a hardcoded True (see main_pipeline.py's result consumer).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    db_path:     str = "ocr_results.db"
    echo_sql:    bool = False


class DatabaseAnalyticsStage:
    """
    Wraps an SQLite database that stores every OCR result and can produce
    analytics reports on demand.

    Thread-safety
    -------------
    A single connection is kept open for the lifetime of the stage; a Lock
    serialises all writes so that multiple threads can call insert_result()
    safely (the result-consumer thread in main_pipeline.py is the only
    writer, but the lock is kept for safety if that changes).
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS ocr_results (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id                 TEXT    NOT NULL,
        track_id               INTEGER NOT NULL,
        job_id                 TEXT    NOT NULL,
        plate_text             TEXT    NOT NULL,
        confidence             REAL    NOT NULL,
        status                 TEXT    NOT NULL,
        is_valid               INTEGER NOT NULL DEFAULT 0,
        finalize_reason        TEXT,
        plate_bbox             TEXT,
        num_readings           INTEGER NOT NULL DEFAULT 0,
        plate_text_bilateral   TEXT,
        conf_bilateral         REAL    DEFAULT 0.0,
        plate_text_adaptive    TEXT,
        conf_adaptive          REAL    DEFAULT 0.0,
        winner_branch          TEXT    DEFAULT 'none',
        stitched_track_ids     TEXT,
        low_diversity          INTEGER NOT NULL DEFAULT 0,
        possible_id_switch     INTEGER NOT NULL DEFAULT 0,
        created_at             DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, config: DatabaseConfig = DatabaseConfig()) -> None:
        self.cfg   = config
        self._lock = threading.Lock()

        Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            config.db_path,
            check_same_thread=False,   # we serialise via _lock
        )
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()
        logger.info("DatabaseAnalyticsStage: opened %s", config.db_path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_result(
        self,
        run_id:               str,
        track_id:             int,
        job_id:               str,
        plate_text:           str,
        confidence:           float,
        status:               str,
        is_valid:             bool  = False,
        finalize_reason:      str   = "",
        plate_bbox:           Optional[tuple] = None,
        num_readings:         int   = 0,
        plate_text_bilateral: str   = "",
        conf_bilateral:       float = 0.0,
        plate_text_adaptive:  str   = "",
        conf_adaptive:        float = 0.0,
        winner_branch:        str   = "none",
        stitched_track_ids:   str   = "",
        low_diversity:        bool  = False,
        possible_id_switch:   bool  = False,
    ) -> None:
        sql = """
            INSERT INTO ocr_results
                (run_id, track_id, job_id, plate_text, confidence,
                 status, is_valid, finalize_reason, plate_bbox,
                 num_readings, plate_text_bilateral, conf_bilateral,
                 plate_text_adaptive, conf_adaptive, winner_branch,
                 stitched_track_ids, low_diversity, possible_id_switch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(sql, (
                run_id, track_id, job_id,
                plate_text, confidence, status,
                int(is_valid), finalize_reason,
                str(plate_bbox) if plate_bbox else None,
                num_readings, plate_text_bilateral, conf_bilateral,
                plate_text_adaptive, conf_adaptive, winner_branch,
                stitched_track_ids, int(low_diversity), int(possible_id_switch),
            ))
            self._conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def fetch_run(self, run_id: str) -> pd.DataFrame:
        sql = "SELECT * FROM ocr_results WHERE run_id = ?"
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=(run_id,))

    def fetch_all(self) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query("SELECT * FROM ocr_results", self._conn)

    # ── Track Stitching ───────────────────────────────────────────────────────

    @staticmethod
    def stitch_fragmented_rows(
        rich_rows: list,
        max_frame_gap: int = 300,
    ) -> tuple[list, dict]:
        """
        Post-tracking trajectory stitcher.
        Groups rich CSV rows by license_number (or Levenshtein edit distance <= 1).
        If multiple track IDs share the same plate within max_frame_gap,
        unifies them under the primary (earliest) track_id.
        Returns (stitched_rows, track_id_mapping).
        """
        if not rich_rows:
            return rich_rows, {}

        from plate_utils import levenshtein_dist

        # Map track_ids to their plate text and frame ranges
        track_info: dict = {}
        for r in rich_rows:
            tid = int(float(r["car_id"]))
            fn  = int(r["frame_nmr"])
            txt = str(r["license_number"]).strip().upper()
            if tid not in track_info:
                track_info[tid] = {"plates": {}, "min_frame": fn, "max_frame": fn}
            track_info[tid]["min_frame"] = min(track_info[tid]["min_frame"], fn)
            track_info[tid]["max_frame"] = max(track_info[tid]["max_frame"], fn)
            track_info[tid]["plates"][txt] = track_info[tid]["plates"].get(txt, 0) + 1

        # Determine dominant plate per track
        for tid, info in track_info.items():
            info["dominant_plate"] = max(info["plates"], key=info["plates"].get)

        # Build alias mapping: secondary_tid -> primary_tid
        alias_map: dict = {}
        sorted_tids = sorted(track_info.keys(), key=lambda t: track_info[t]["min_frame"])

        for i, tid1 in enumerate(sorted_tids):
            primary = alias_map.get(tid1, tid1)
            p1 = track_info[tid1]["dominant_plate"]
            if not p1 or p1 == "NAN":
                continue

            for tid2 in sorted_tids[i+1:]:
                if tid2 in alias_map:
                    continue
                p2 = track_info[tid2]["dominant_plate"]
                if not p2 or p2 == "NAN":
                    continue

                # Check plate similarity and temporal gap
                dist = levenshtein_dist(p1, p2)
                frame_gap = track_info[tid2]["min_frame"] - track_info[tid1]["max_frame"]

                if dist <= 1 and 0 <= frame_gap <= max_frame_gap:
                    alias_map[tid2] = primary
                    logger.info(
                        "Stitching Track %d into Track %d (Plate '%s' ~ '%s', gap=%d frames)",
                        tid2, primary, p1, p2, frame_gap,
                    )

        # Apply alias mapping to rows
        stitched_rows = []
        for r in rich_rows:
            row_copy = dict(r)
            orig_tid = int(float(row_copy["car_id"]))
            if orig_tid in alias_map:
                row_copy["car_id"] = alias_map[orig_tid]
            stitched_rows.append(row_copy)

        return stitched_rows, alias_map

    # ── Analytics ─────────────────────────────────────────────────────────────

    def generate_report(self, run_id: Optional[str] = None, output_dir: str = ".") -> str:
        df = self.fetch_run(run_id) if run_id else self.fetch_all()

        if df.empty:
            logger.warning("generate_report: no data to plot")
            return ""

        df["created_at"] = pd.to_datetime(df["created_at"])

        # 4-panel analytical summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle(
            f"ALPR Pipeline Analytics & Preprocessing Comparison{f' — run {run_id}' if run_id else ''}",
            fontsize=14, fontweight="bold",
        )

        # Panel 1: Recognition Status Breakdown
        status_counts = df["status"].value_counts()
        axes[0, 0].bar(status_counts.index, status_counts.values, color=["forestgreen", "crimson", "orange"])
        axes[0, 0].set_title("Recognition Status Breakdown", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Count")

        # Panel 2: Preprocessing Comparison (Bilateral vs Adaptive)
        success_df = df[df["status"] == "SUCCESS"].copy()
        if not success_df.empty and "winner_branch" in success_df.columns:
            winner_counts = success_df["winner_branch"].value_counts()
            axes[0, 1].pie(
                winner_counts.values,
                labels=winner_counts.index,
                autopct="%1.1f%%",
                colors=["#3498db", "#f39c12", "#2ecc71", "#95a5a6"],
                startangle=140,
            )
            axes[0, 1].set_title("Preprocessing Branch Winner Share (Bilateral vs Adaptive)", fontsize=11, fontweight="bold")
        else:
            axes[0, 1].text(0.5, 0.5, "No Success Data", ha="center", va="center")

        # Panel 3: Confidence Comparison Boxplot / Histogram
        if not success_df.empty:
            b_confs = success_df["conf_bilateral"].dropna()
            a_confs = success_df["conf_adaptive"].dropna()
            axes[1, 0].hist(b_confs, bins=15, alpha=0.6, label="Bilateral", color="steelblue")
            axes[1, 0].hist(a_confs, bins=15, alpha=0.6, label="Adaptive", color="darkorange")
            axes[1, 0].set_title("Confidence Distribution by Preprocessing Branch", fontsize=11, fontweight="bold")
            axes[1, 0].set_xlabel("Confidence Score")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
        else:
            axes[1, 0].set_title("Confidence Distribution")

        # Panel 4: Top 10 Recognized Plates
        if not success_df.empty:
            top_plates = (
                success_df.groupby("plate_text")["confidence"]
                .max()
                .sort_values(ascending=False)
                .head(10)
            )
            axes[1, 1].barh(top_plates.index, top_plates.values, color="teal")
            axes[1, 1].set_title("Top 10 Fused Plates by Confidence", fontsize=11, fontweight="bold")
            axes[1, 1].set_xlabel("Confidence")

        plt.tight_layout()
        out_path = str(Path(output_dir) / f"analytics{'_' + run_id if run_id else ''}.png")
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        logger.info("Analytics report saved -> %s", out_path)

        # Also write comparison CSV
        if not df.empty:
            comp_path = str(Path(output_dir) / "results_preprocessing_comparison.csv")
            comp_cols = [c for c in [
                "track_id", "status", "plate_text", "confidence",
                "plate_text_bilateral", "conf_bilateral",
                "plate_text_adaptive", "conf_adaptive",
                "winner_branch", "num_readings", "stitched_track_ids",
            ] if c in df.columns]
            df[comp_cols].to_csv(comp_path, index=False)
            logger.info("Preprocessing comparison CSV saved -> %s", comp_path)

        return out_path

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("DatabaseAnalyticsStage: connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
