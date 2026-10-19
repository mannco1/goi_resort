"""Silero VAD v5 через onnxruntime, покадрово (512 сэмплов при 16 кГц)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as rt

VAD_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
CONTEXT = 64  # Silero v5 ждёт 64 сэмпла предыдущего кадра перед текущим


def ensure_vad_model(models_dir: Path) -> Path:
    path = models_dir / "silero_vad.onnx"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print("[vad] скачиваю Silero VAD…")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(VAD_URL, tmp)
        tmp.rename(path)
    return path


class SileroVAD:
    def __init__(self, model_path: Path):
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = rt.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sr = np.array(16_000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = x[:, -CONTEXT:]
        return float(out[0, 0])
