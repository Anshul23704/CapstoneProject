from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameIngestionConfig:
    source: str | int
    target_fps: Optional[float] = None
    target_resolution: Optional[Tuple[int, int]] = None
    max_retries: int = 3
    backend: int = cv2.CAP_ANY
    queue_full_sleep_ms: int = 1


class FrameIngestionStage:

    def __init__(self, config: FrameIngestionConfig) -> None:
        self.cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.dropped_frames: int = 0
        self._fps_samples: list[float] = []

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.cfg.source, self.cfg.backend)

        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video source: {self.cfg.source}"
            )

        if self.cfg.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)

        if self.cfg.target_resolution:
            w, h = self.cfg.target_resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def close(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def frames(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:

        if self._cap is None:
            self.open()

        consecutive_failures = 0
        frame_idx = 0
        t_prev = time.monotonic()

        while self._cap.isOpened():

            ret, frame = self._cap.read()
            ts = time.monotonic()

            if not ret:
                consecutive_failures += 1

                if consecutive_failures >= self.cfg.max_retries:
                    break

                time.sleep(0.005 * consecutive_failures)
                continue

            consecutive_failures = 0

            if frame is None or frame.size == 0:
                self.dropped_frames += 1
                continue

            if self.cfg.target_resolution:
                w, h = self.cfg.target_resolution
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))

            elapsed = ts - t_prev
            if elapsed > 0:
                self._fps_samples.append(1.0 / elapsed)

                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)

            t_prev = ts

            self.total_frames += 1

            yield frame, frame_idx, ts

            frame_idx += 1

    @property
    def actual_fps(self) -> float:
        if not self._fps_samples:
            return 0.0
        return sum(self._fps_samples) / len(self._fps_samples)

    @property
    def source_frame_count(self) -> int:
        """
        Frame count reported by the container/codec (may be an estimate for
        some formats). Public accessor — previously callers (main_pipeline.py)
        reached into the private `_cap` attribute directly, which breaks
        encapsulation and crashes if called before open().
        """
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
