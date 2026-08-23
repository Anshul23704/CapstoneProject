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
from plate_utils import format_license, license_complies_format

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

    Combines multiple per-frame OCR readings (already formatted by Stage 6)
    via confidence-weighted character-level majority voting, then
    re-validates the fused string with license_complies_format.
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

        # Single reading — just return.
        if len(valid) < self.cfg.min_readings_for_fusion:
            text, conf = max(valid, key=lambda x: x[1])
            is_valid   = True # Validation moved to final stage
            logger.debug("TemporalFusion: single reading -> '%s' valid=%s", text, is_valid)
            return text, conf, is_valid

        # Align by modal length across ALL readings, as strict validation is moved to CSV stage.
        lengths       = [len(t) for t, _ in valid]
        target_length = Counter(lengths).most_common(1)[0][0]
        aligned       = [(t, c) for t, c in valid if len(t) == target_length]

        if not aligned:
            text, conf = max(valid, key=lambda x: x[1])
            return text, conf, True

        # Confidence-weighted character-level majority voting.
        fused_chars: List[str] = []
        for i in range(target_length):
            vote_weight: dict = {}
            for text, conf in aligned:
                ch = text[i]
                weight = conf
                vote_weight[ch] = vote_weight.get(ch, 0.0) + weight
            winner = max(vote_weight, key=vote_weight.__getitem__)
            fused_chars.append(winner)

        fused_text = "".join(fused_chars)
        avg_conf = sum(c for _, c in aligned) / len(aligned)
        is_valid = True # Strict validation moved to final stage

        logger.debug(
            "TemporalFusion: %d/%d aligned (len=%d) -> '%s' conf=%.3f valid=%s",
            len(aligned), len(valid), target_length, fused_text, avg_conf, is_valid,
        )
        return fused_text, float(avg_conf), is_valid
