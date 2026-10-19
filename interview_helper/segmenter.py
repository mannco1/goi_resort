"""Нарезка потока на фразы по VAD.

События:
  start   — человек начал говорить
  partial — фраза ещё идёт, вот весь звук с начала (для «живого» текста)
  final   — фраза закончилась
  discard — это был короткий шум, а не речь
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .audio import FRAME, SR
from .config import VadCfg
from .vad import SileroVAD

FRAME_MS = FRAME * 1000 / SR  # 32 мс


@dataclass
class SegEvent:
    kind: str            # start | partial | final | discard
    source: str          # interviewer | me
    seg_id: int
    audio: np.ndarray | None = None
    t: float = 0.0       # время события (time.time())


class Segmenter:
    START_FRAMES = 3     # ~100 мс уверенной речи, чтобы начать фразу

    def __init__(self, source: str, vad: SileroVAD, cfg: VadCfg,
                 on_event: Callable[[SegEvent], None], partials: bool = True):
        self.source = source
        self.vad = vad
        self.cfg = cfg
        self.on_event = on_event
        self.partials = partials
        self.silence_frames = max(1, int(cfg.silence_ms / FRAME_MS))
        self.min_frames = max(1, int(cfg.min_speech_ms / FRAME_MS))
        self.max_frames = int(cfg.max_segment_s * 1000 / FRAME_MS)
        self.partial_frames = max(1, int(cfg.partial_interval_s * 1000 / FRAME_MS))
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.preroll_ms / FRAME_MS)))
        self.end_threshold = max(0.05, cfg.threshold - 0.15)  # гистерезис: выйти из речи сложнее, чем войти
        self.paused = False
        self.seg_id = 0
        self._reset()

    def _reset(self) -> None:
        self.in_speech = False
        self.buf: list[np.ndarray] = []
        self.run = 0
        self.silence = 0
        self.voiced = 0
        self.since_partial = 0

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused and self.in_speech:
            self._finish(force_discard=True)
        self.vad.reset()
        self.preroll.clear()

    def feed(self, frame: np.ndarray) -> None:
        if self.paused:
            return
        p = self.vad(frame)
        if not self.in_speech:
            self.preroll.append(frame)
            self.run = self.run + 1 if p >= self.cfg.threshold else 0
            if self.run >= self.START_FRAMES:
                self._begin(list(self.preroll))
            return

        self.buf.append(frame)
        if p < self.end_threshold:
            self.silence += 1
        else:
            self.silence = 0
            self.voiced += 1
        self.since_partial += 1

        if self.silence >= self.silence_frames:
            self._finish()
        elif len(self.buf) >= self.max_frames:
            self._finish()
            self._begin([])          # человек всё ещё говорит — сразу открываем следующую фразу
        elif self.partials and self.since_partial >= self.partial_frames:
            self.since_partial = 0
            self._emit("partial", np.concatenate(self.buf))

    def _begin(self, initial: list[np.ndarray]) -> None:
        self.seg_id += 1
        self._reset()
        self.in_speech = True
        self.buf = initial
        self.voiced = self.START_FRAMES if initial else 0
        self.preroll.clear()
        self._emit("start")

    def _finish(self, force_discard: bool = False) -> None:
        # оставляем ~200 мс хвостовой тишины, остальное отрезаем
        keep_tail = int(200 / FRAME_MS)
        trim = max(0, self.silence - keep_tail)
        frames = self.buf[:len(self.buf) - trim] if trim else self.buf
        if force_discard or self.voiced < self.min_frames or not frames:
            self._emit("discard")
        else:
            self._emit("final", np.concatenate(frames))
        self._reset()

    def _emit(self, kind: str, audio: np.ndarray | None = None) -> None:
        self.on_event(SegEvent(kind, self.source, self.seg_id, audio, time.time()))
