"""
Stage 8 — Database Storage & Analytics.

BUG FIXES vs previous version
──────────────────────────────
1. Previous file executed INSERT + matplotlib at import time — completely broken
   when imported by main_pipeline.py.  Now everything is wrapped in a class.
2. main_pipeline.py imported `DatabaseAnalyticsStage` and `DatabaseConfig` but
   neither existed in the old file.  Both are now defined properly.
3. The table schema was too narrow (only `result TEXT`).  Expanded to store all
   useful recognition fields for proper analytics.
4. Connection was never thread-safe; added a threading.Lock so worker threads
   can insert concurrently without "database is locked" errors.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    serialises all writes so that multiple worker threads can call
    insert_result() safely.
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS ocr_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT    NOT NULL,
        track_id      INTEGER NOT NULL,
        job_id        TEXT    NOT NULL,
        plate_text    TEXT    NOT NULL,
        confidence    REAL    NOT NULL,
        status        TEXT    NOT NULL,
        is_valid      INTEGER NOT NULL DEFAULT 0,
        finalize_reason TEXT,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, config: DatabaseConfig = DatabaseConfig()) -> None:
        self.cfg   = config
        self._lock = threading.Lock()

        # Ensure parent directory exists
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
        run_id:          str,
        track_id:        int,
        job_id:          str,
        plate_text:      str,
        confidence:      float,
        status:          str,
        is_valid:        bool  = False,
        finalize_reason: str   = "",
    ) -> None:
        sql = """
            INSERT INTO ocr_results
                (run_id, track_id, job_id, plate_text, confidence,
                 status, is_valid, finalize_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(sql, (
                run_id, track_id, job_id,
                plate_text, confidence, status,
                int(is_valid), finalize_reason,
            ))
            self._conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def fetch_run(self, run_id: str) -> pd.DataFrame:
        """Return all rows for a specific pipeline run."""
        sql = "SELECT * FROM ocr_results WHERE run_id = ?"
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=(run_id,))

    def fetch_all(self) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query("SELECT * FROM ocr_results", self._conn)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def generate_report(self, run_id: Optional[str] = None, output_dir: str = ".") -> str:
        """
        Produce a PNG analytics report.

        BUG FIX: old version always plotted ALL rows regardless of run.
        Now scoped to a specific run_id when supplied.

        Returns the path of the saved PNG.
        """
        df = self.fetch_run(run_id) if run_id else self.fetch_all()

        if df.empty:
            logger.warning("generate_report: no data to plot")
            return ""

        df["created_at"] = pd.to_datetime(df["created_at"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(
            f"ALPR Pipeline Analytics{f' — run {run_id}' if run_id else ''}",
            fontsize=14,
        )

        # Plot 1: status breakdown
        status_counts = df["status"].value_counts()
        axes[0, 0].bar(status_counts.index, status_counts.values, color=["green", "red", "orange"])
        axes[0, 0].set_title("Recognition Status Breakdown")
        axes[0, 0].set_ylabel("Count")

        # Plot 2: confidence distribution (successes only)
        success_df = df[df["status"] == "SUCCESS"]
        if not success_df.empty:
            axes[0, 1].hist(success_df["confidence"], bins=20, color="steelblue", edgecolor="black")
        axes[0, 1].set_title("Confidence Distribution (Successes)")
        axes[0, 1].set_xlabel("Confidence")

        # Plot 3: results over time (by hour)
        df_time = df.set_index("created_at").resample("h").size()
        axes[1, 0].plot(df_time.index, df_time.values, marker="o")
        axes[1, 0].set_title("Results per Hour")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].tick_params(axis="x", rotation=30)

        # Plot 4: top 10 unique plates
        if not success_df.empty:
            top_plates = (
                success_df.groupby("plate_text")["confidence"]
                .max()
                .sort_values(ascending=False)
                .head(10)
            )
            axes[1, 1].barh(top_plates.index, top_plates.values, color="teal")
            axes[1, 1].set_title("Top 10 Plates by Confidence")
            axes[1, 1].set_xlabel("Confidence")

        plt.tight_layout()
        out_path = str(Path(output_dir) / f"analytics{'_' + run_id if run_id else ''}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Analytics report saved → %s", out_path)
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
