from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """
    Configuration for the Temporal Fusion stage.

    regex_pattern
        A compiled-regex string used to validate the fused plate string.
        Swap this out for a country-specific pattern as needed:

        Generic  (default) : r"^[A-Z0-9]{4,10}$"
        UK                 : r"^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$"
        India              : r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"

    min_readings_for_fusion
        Minimum number of non-empty OCR readings required before voting is
        attempted.  If fewer readings arrive, the highest-confidence single
        reading is returned directly (with its validation result).
    """
    regex_pattern:            str = r"^[A-Z0-9]{4,10}$"
    min_readings_for_fusion:  int = 2


class TemporalFusionStage:
    """
    Stage 7 — Temporal Fusion & Validation.

    Combines multiple per-frame OCR readings into a single reliable plate
    string using character-level majority voting, then validates the result
    against a configurable regex pattern.

    Why this matters
    ----------------
    EasyOCR is frequently over-confident on individual characters that are
    ambiguous in a single frame (e.g. B↔8, 1↔I, O↔0).  By aggregating
    readings across 3-5 frames and voting at each character position, the
    true plate string can be reconstructed even when no single frame was
    100% correct.

    Usage
    -----
        stage = TemporalFusionStage(FusionConfig())
        fused_text, avg_conf, is_valid = stage.process(frame_readings)

    Where frame_readings is a list of (text: str, confidence: float) tuples
    produced by the Stage 6 OCR step, one tuple per selected frame.
    """

    def __init__(self, config: FusionConfig = FusionConfig()) -> None:
        self.cfg       = config
        self.validator = re.compile(self.cfg.regex_pattern)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        readings: List[Tuple[str, float]],
    ) -> Tuple[str, float, bool]:
        """
        Parameters
        ----------
        readings : list of (text, confidence) tuples
            Raw OCR output collected across all selected frames for one job.
            Empty strings and whitespace-only strings are filtered out.

        Returns
        -------
        fused_text : str
            The majority-voted plate string (uppercase, no spaces or hyphens).
        avg_confidence : float
            Mean confidence of the readings that were aligned and voted on.
        is_valid : bool
            True if fused_text matches the configured regex_pattern.
        """
        # ── 1. Clean & normalise ──────────────────────────────────────────
        # Strip spaces/hyphens that EasyOCR sometimes inserts, force upper.
        valid_readings: List[Tuple[str, float]] = [
            (text.upper().replace(" ", "").replace("-", ""), conf)
            for text, conf in readings
            if text and text.strip()
        ]

        if not valid_readings:
            logger.debug("TemporalFusion: no valid readings to fuse")
            return "", 0.0, False

        # ── 2. Fallback: not enough readings for meaningful voting ────────
        if len(valid_readings) < self.cfg.min_readings_for_fusion:
            best_text, best_conf = max(valid_readings, key=lambda x: x[1])
            is_valid = bool(self.validator.match(best_text))
            logger.debug(
                "TemporalFusion: only %d reading(s) — using best directly: '%s' (valid=%s)",
                len(valid_readings), best_text, is_valid,
            )
            return best_text, best_conf, is_valid

        # ── 3. Align by modal string length ──────────────────────────────
        # Different frames may produce strings of different lengths when OCR
        # picks up an extra spurious character (e.g. "AB12CDE" vs "B12CDE").
        # We find the most common length and restrict voting to those readings,
        # so that character positions are correctly aligned.
        lengths       = [len(text) for text, _ in valid_readings]
        target_length = Counter(lengths).most_common(1)[0][0]

        aligned_readings = [
            (text, conf)
            for text, conf in valid_readings
            if len(text) == target_length
        ]

        if not aligned_readings:
            # Shouldn't happen, but guard defensively
            best_text, best_conf = max(valid_readings, key=lambda x: x[1])
            is_valid = bool(self.validator.match(best_text))
            logger.debug(
                "TemporalFusion: alignment produced empty set — fallback: '%s'", best_text
            )
            return best_text, best_conf, is_valid

        # ── 4. Character-level majority voting ────────────────────────────
        # For each position i, collect the character voted by every aligned
        # reading and pick the plurality winner.
        fused_chars: List[str] = []
        for i in range(target_length):
            char_votes = [text[i] for text, _ in aligned_readings]
            winner, _  = Counter(char_votes).most_common(1)[0]
            fused_chars.append(winner)

        fused_text = "".join(fused_chars)

        # ── 5. Aggregate confidence ───────────────────────────────────────
        avg_conf = sum(conf for _, conf in aligned_readings) / len(aligned_readings)

        # ── 6. Format validation ──────────────────────────────────────────
        is_valid = bool(self.validator.match(fused_text))

        logger.debug(
            "TemporalFusion: %d/%d readings aligned (len=%d) → '%s' conf=%.3f valid=%s",
            len(aligned_readings), len(valid_readings),
            target_length, fused_text, avg_conf, is_valid,
        )

        return fused_text, float(avg_conf), is_valid
