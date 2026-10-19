"""Захват звука: WASAPI loopback (то, что слышите вы) и микрофон.

Всё приводится к 16 кГц, моно, float32 и отдаётся кадрами по 512 сэмплов (32 мс) —
это размер окна, на котором работает Silero VAD.
"""
from __future__ import annotations

import queue
import threading
import time
import wave
from typing import Callable

import numpy as np
import soxr

SR = 16_000
FRAME = 512  # 32 мс при 16 кГц

FrameCallback = Callable[[np.ndarray], None]


class _Framer:
    """Ресемплинг потока в 16 кГц моно + нарезка на кадры по FRAME сэмплов."""

    def __init__(self, in_rate: int, channels: int, on_frame: FrameCallback):
        self.channels = channels
        self.on_frame = on_frame
        self.rs = soxr.ResampleStream(in_rate, SR, 1, dtype="float32", quality="HQ") if in_rate != SR else None
        self.buf = np.zeros(0, dtype=np.float32)

    def push_int16(self, raw: bytes) -> None:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self.push_float(x)

    def push_float(self, x: np.ndarray) -> None:
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        if self.rs is not None:
            x = self.rs.resample_chunk(x)
        if x.size == 0:
            return
        self.buf = np.concatenate([self.buf, x.astype(np.float32, copy=False)])
        n = self.buf.size // FRAME
        for i in range(n):
            self.on_frame(self.buf[i * FRAME:(i + 1) * FRAME])
        self.buf = self.buf[n * FRAME:]


class _Worker:
    """Колбэк PortAudio кладёт сырые байты в очередь, тяжёлую работу делает отдельный поток."""

    def __init__(self, framer: _Framer, name: str):
        self.framer = framer
        self.q: queue.Queue[bytes | None] = queue.Queue(maxsize=500)
        self.thread = threading.Thread(target=self._run, name=f"audio-{name}", daemon=True)
        self.thread.start()

    def put(self, raw: bytes) -> None:
        try:
            self.q.put_nowait(raw)
        except queue.Full:  # если обработка не успевает — лучше потерять кусок, чем копить задержку
            pass

    def stop(self) -> None:
        self.q.put(None)

    def _run(self) -> None:
        while (raw := self.q.get()) is not None:
            self.framer.push_int16(raw)


class WasapiSource:
    """Источник звука на Windows через PyAudioWPatch (форк PyAudio с поддержкой loopback)."""

    def __init__(self, kind: str, on_frame: FrameCallback, device_hint: str = ""):
        import pyaudiowpatch as pyaudio  # импорт здесь, чтобы остальное работало и без Windows

        self.pa_mod = pyaudio
        self.pa = pyaudio.PyAudio()
        self.kind = kind
        dev = self._find_device(kind, device_hint)
        self.device_name = dev["name"]
        channels = int(dev["maxInputChannels"]) or 2
        rate = int(dev["defaultSampleRate"])
        self.worker = _Worker(_Framer(rate, channels, on_frame), kind)

        def cb(in_data, frame_count, time_info, status):
            self.worker.put(in_data)
            return (None, pyaudio.paContinue)

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(dev["index"]),
            frames_per_buffer=int(rate * 0.03),
            stream_callback=cb,
        )

    def _find_device(self, kind: str, hint: str) -> dict:
        pa = self.pa
        wasapi = pa.get_host_api_info_by_type(self.pa_mod.paWASAPI)
        hint_l = hint.lower()
        if kind == "system":
            loopbacks = list(pa.get_loopback_device_info_generator())
            if hint_l:
                for d in loopbacks:
                    if hint_l in d["name"].lower():
                        return d
                print(f"[audio] loopback-устройство «{hint}» не найдено, беру устройство по умолчанию")
            speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if speakers.get("isLoopbackDevice"):
                return speakers
            for d in loopbacks:
                if speakers["name"] in d["name"]:
                    return d
            if loopbacks:
                return loopbacks[0]
            raise RuntimeError("Не нашёл WASAPI loopback-устройство. Проверьте, что выбрано устройство вывода звука.")
        # микрофон
        if hint_l:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if (d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0
                        and not d.get("isLoopbackDevice") and hint_l in d["name"].lower()):
                    return d
            print(f"[audio] микрофон «{hint}» не найден, беру микрофон по умолчанию")
        idx = wasapi["defaultInputDevice"]
        if idx < 0:
            raise RuntimeError("Не нашёл микрофон по умолчанию.")
        return pa.get_device_info_by_index(idx)

    def start(self) -> None:
        self.stream.start_stream()

    def stop(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.worker.stop()
            self.pa.terminate()


class WavFileSource:
    """Проигрывает wav-файл в конвейер так, как будто это звук собеседника (для отладки без звонка)."""

    def __init__(self, path: str, on_frame: FrameCallback, speed: float = 1.0):
        self.path = path
        self.speed = speed
        self.on_frame = on_frame
        self.device_name = f"файл {path}"
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="audio-file", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        with wave.open(self.path, "rb") as w:
            if w.getsampwidth() != 2:
                raise RuntimeError("Поддерживаются только 16-битные PCM wav-файлы")
            rate, ch = w.getframerate(), w.getnchannels()
            framer = _Framer(rate, ch, self.on_frame)
            block = int(rate * 0.05)
            while not self._stop.is_set():
                raw = w.readframes(block)
                if not raw:
                    break
                framer.push_int16(raw)
                time.sleep(0.05 / self.speed)
        # хвост тишины, чтобы последняя фраза закрылась
        silence = np.zeros(FRAME, dtype=np.float32)
        for _ in range(60):
            if self._stop.is_set():
                break
            self.on_frame(silence)
            time.sleep(FRAME / SR / self.speed)


def list_devices() -> None:
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        print("Loopback (звук собеседника) — укажите часть имени в audio.system_device:")
        for d in pa.get_loopback_device_info_generator():
            print(f"   [{d['index']}] {d['name']}  {int(d['defaultSampleRate'])} Гц")
        print("\nМикрофоны (WASAPI) — укажите часть имени в audio.mic_device:")
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice"):
                mark = " (по умолчанию)" if i == wasapi["defaultInputDevice"] else ""
                print(f"   [{i}] {d['name']}{mark}")
    finally:
        pa.terminate()
