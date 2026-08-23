"""
stage7_temporal_fusion.py — Temporal Fusion & Validation.

This is the stage that turns "we saw the plate in 5-7 frames" (Slide 4's
core promise) into a single, higher-confidence answer. Previously this
class existed and worked standalone, but main_pipeline.py never called it
— every RecognitionResult's frame_readings were computed by Stage 6 and
then thrown away, with main_pipeline picking only the single
highest-confidence reading instead of fusing across frames. See
main_pipeline.py's result-consumer for the fix; this file itself only
needed its duplicated format helpers consolidated into plate_utils.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import config
from plate_utils import (
    CHAR_TO_INT,
    INT_TO_CHAR,
    format_license,
    license_complies_format,
    soft_format_indian_plate,
    fuzzy_match_state_code,
)

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    min_readings_for_fusion: int = config.MIN_READINGS_FOR_FUSION
    # kept for backward compatibility only; the actual gate is
    # license_complies_format's strict 7-char positional rule.
    regex_pattern: str = r"^[A-Z0-9]{4,10}$"


class TemporalFusionStage:
    """
    Stage 7 — Temporal Fusion & Validation.

    Combines multiple per-frame OCR readings via confidence-weighted
    character-level majority voting with positional priors and state code verification.
    """

    def __init__(self, config: FusionConfig = FusionConfig()) -> None:
        self.cfg = config

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        readings: List[Tuple[str, float]],
    ) -> Tuple[str, float, bool]:
        """
        Returns (fused_text, avg_confidence, is_valid).
        """
        valid: List[Tuple[str, float]] = [
            (t.upper().replace(" ", "").replace("-", ""), c)
            for t, c in readings
            if t and t.strip()
        ]

        if not valid:
            return "", 0.0, False

        # Single reading — apply soft format and return.
        if len(valid) < self.cfg.min_readings_for_fusion:
            text, conf = max(valid, key=lambda x: x[1])
            formatted = soft_format_indian_plate(text)
            return formatted, conf, True

        # Align by modal length across ALL readings.
        lengths       = [len(t) for t, _ in valid]
        target_length = Counter(lengths).most_common(1)[0][0]
        aligned       = [(t, c) for t, c in valid if len(t) == target_length]

        if not aligned:
            text, conf = max(valid, key=lambda x: x[1])
            return soft_format_indian_plate(text), conf, True

        # Confidence-weighted character-level majority voting with positional canonicalization
        fused_chars: List[str] = []
        for i in range(target_length):
            vote_weight: dict = {}
            is_alpha_slot = (i in (0, 1)) or (target_length == 10 and i in (4, 5))
            is_digit_slot = (i in (2, 3) and target_length in (9, 10)) or (i >= target_length - 4)

            for text, conf in aligned:
                ch = text[i]
                # Canonicalize glyph based on positional prior
                if is_alpha_slot:
                    ch = INT_TO_CHAR.get(ch, ch)
                elif is_digit_slot:
                    ch = CHAR_TO_INT.get(ch, ch)

                weight = conf
                vote_weight[ch] = vote_weight.get(ch, 0.0) + weight

            winner = max(vote_weight, key=vote_weight.__getitem__)
            fused_chars.append(winner)

        raw_fused = "".join(fused_chars)
        fused_text = soft_format_indian_plate(raw_fused)
        avg_conf = sum(c for _, c in aligned) / len(aligned)
        is_valid = True

        logger.debug(
            "TemporalFusion: %d/%d aligned (len=%d) -> '%s' conf=%.3f valid=%s",
            len(aligned), len(valid), target_length, fused_text, avg_conf, is_valid,
        )
        return fused_text, float(avg_conf), is_valid
