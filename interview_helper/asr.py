"""Распознавание речи на NVIDIA Parakeet TDT 0.6B v3 (ONNX, через onnx-asr).

Один поток-воркер на оба источника. Финальные фразы обрабатываются в первую очередь,
промежуточные («живой» текст) — только самые свежие, устаревшие выкидываются.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .audio import SR
from .config import AsrCfg
from .segmenter import SegEvent


@dataclass
class AsrResult:
    kind: str        # partial | final
    source: str
    seg_id: int
    text: str
    t: float
    took: float      # сколько секунд заняло распознавание


class ParakeetASR:
    def __init__(self, cfg: AsrCfg):
        import onnx_asr

        kwargs = {}
        if cfg.providers:
            kwargs["providers"] = cfg.providers
        path = cfg.model_dir or None
        quant = cfg.quantization or None
        try:
            self.model = onnx_asr.load_model(cfg.model, path, quantization=quant, **kwargs)
        except Exception as e:
            if quant is None:
                raise
            print(f"[asr] не удалось загрузить квантованную ({quant}) модель: {e}\n[asr] пробую полную точность…")
            self.model = onnx_asr.load_model(cfg.model, path, **kwargs)
        self.recognize(np.zeros(SR, dtype=np.float32))  # прогрев, чтобы первая фраза не тормозила

    def recognize(self, audio: np.ndarray) -> str:
        return str(self.model.recognize(audio, sample_rate=SR)).strip()


class AsrWorker:
    def __init__(self, asr, on_result: Callable[[AsrResult], None]):
        self.asr = asr
        self.on_result = on_result
        self.finals: deque[SegEvent] = deque()
        self.partials: dict[str, SegEvent] = {}
        self.cv = threading.Condition()
        self._stop = False
        self.thread = threading.Thread(target=self._run, name="asr", daemon=True)
        self.thread.start()

    def submit(self, ev: SegEvent) -> None:
        with self.cv:
            if ev.kind == "final":
                self.finals.append(ev)
                # промежуточный кусок той же фразы больше не нужен
                p = self.partials.get(ev.source)
                if p and p.seg_id == ev.seg_id:
                    del self.partials[ev.source]
            elif ev.kind == "partial":
                self.partials[ev.source] = ev
            else:
                return
            self.cv.notify()

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify()

    def _next(self) -> SegEvent | None:
        with self.cv:
            while not self._stop and not self.finals and not self.partials:
                self.cv.wait()
            if self._stop:
                return None
            if self.finals:
                return self.finals.popleft()
            src = next(iter(self.partials))
            return self.partials.pop(src)

    def _run(self) -> None:
        while (ev := self._next()) is not None:
            t0 = time.perf_counter()
            try:
                text = self.asr.recognize(ev.audio)
            except Exception as e:  # noqa: BLE001 — одна неудачная фраза не должна ронять приложение
                print(f"[asr] ошибка распознавания: {e}")
                continue
            self.on_result(AsrResult(ev.kind, ev.source, ev.seg_id, text, ev.t, time.perf_counter() - t0))
