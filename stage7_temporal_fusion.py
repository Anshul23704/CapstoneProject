"""
stage7_temporal_fusion.py — Temporal Fusion & Validation

INTEGRATION CHANGES (vs previous version)
──────────────────────────────────────────
1. frame_readings arriving from Stage 6 are now already validated and
   formatted (license_complies_format + format_license were applied in
   the worker). Stage 7 no longer needs to re-validate junk; it just
   needs to pick the best reading or fuse multiple readings.

2. The validation step at the end still applies license_complies_format
   so that the fused string (which may have been character-voted into a
   new shape) is re-checked before being marked VALID.

3. FusionConfig.regex_pattern is kept for backward compatibility but is
   no longer the primary gate — license_complies_format is.

4. _apply_confusion_fix is kept as a second-pass safety net but is now
   a thin wrapper around format_license so both pipelines use the same
   correction table.
"""

from __future__ import annotations

import logging
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ── Shared format helpers (same as stage6) ────────────────────────────────────

_CHAR_TO_INT = {
    'O': '0', 'I': '1', 'J': '3',
    'A': '4', 'G': '6', 'S': '5',
}
_INT_TO_CHAR = {v: k for k, v in _CHAR_TO_INT.items()}


def license_complies_format(text: str) -> bool:
    if len(text) != 7:
        return False
    alpha_ok = set(string.ascii_uppercase) | set(_INT_TO_CHAR.keys())
    digit_ok  = set('0123456789')          | set(_CHAR_TO_INT.keys())
    checks = [
        text[0] in alpha_ok,
        text[1] in alpha_ok,
        text[2] in digit_ok,
        text[3] in digit_ok,
        text[4] in alpha_ok,
        text[5] in alpha_ok,
        text[6] in alpha_ok,
    ]
    return all(checks)


def format_license(text: str) -> str:
    mapping = {
        0: _INT_TO_CHAR, 1: _INT_TO_CHAR,
        2: _CHAR_TO_INT, 3: _CHAR_TO_INT,
        4: _INT_TO_CHAR, 5: _INT_TO_CHAR, 6: _INT_TO_CHAR,
    }
    out = []
    for j in range(len(text)):
        ch = text[j]
        out.append(mapping.get(j, {}).get(ch, ch))
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionConfig:
    min_readings_for_fusion: int = 2
    # regex_pattern kept for backward compatibility only; actual gate is
    # license_complies_format which enforces strict 7-char positional rules.
    regex_pattern: str = r"^[A-Z0-9]{4,10}$"


class TemporalFusionStage:
    """
    Stage 7 — Temporal Fusion & Validation.

    Combines multiple per-frame OCR readings (already formatted by Stage 6)
    via confidence-weighted character-level majority voting, then re-validates
    the fused string with license_complies_format.
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
        # 1. Clean & normalise (readings from Stage 6 are already uppercase/stripped
        #    but we sanitise again for safety)
        valid: List[Tuple[str, float]] = [
            (t.upper().replace(" ", "").replace("-", ""), c)
            for t, c in readings
            if t and t.strip()
        ]

        if not valid:
            return "", 0.0, False

        # 2. Single reading — just re-validate and return
        if len(valid) < self.cfg.min_readings_for_fusion:
            text, conf = max(valid, key=lambda x: x[1])
            text       = format_license(text) if len(text) == 7 else text
            is_valid   = license_complies_format(text)
            logger.debug("TemporalFusion: single reading → '%s' valid=%s", text, is_valid)
            return text, conf, is_valid

        # 3. Align by modal length
        lengths       = [len(t) for t, _ in valid]
        target_length = Counter(lengths).most_common(1)[0][0]
        aligned       = [(t, c) for t, c in valid if len(t) == target_length]

        if not aligned:
            text, conf = max(valid, key=lambda x: x[1])
            text       = format_license(text) if len(text) == 7 else text
            return text, conf, license_complies_format(text)

        # 4. Confidence-weighted character-level majority voting
        fused_chars: List[str] = []
        for i in range(target_length):
            vote_weight: dict = {}
            for text, conf in aligned:
                ch = text[i]
                vote_weight[ch] = vote_weight.get(ch, 0.0) + conf
            winner = max(vote_weight, key=vote_weight.__getitem__)
            fused_chars.append(winner)

        fused_text = "".join(fused_chars)

        # 5. Aggregate confidence
        avg_conf = sum(c for _, c in aligned) / len(aligned)

        # 6. Apply positional correction on the fused result
        if len(fused_text) == 7:
            fused_text = format_license(fused_text)

        # 7. Validate with the strict format check
        is_valid = license_complies_format(fused_text)

        logger.debug(
            "TemporalFusion: %d/%d aligned (len=%d) → '%s' conf=%.3f valid=%s",
            len(aligned), len(valid), target_length, fused_text, avg_conf, is_valid,
        )
        return fused_text, float(avg_conf), is_valid
