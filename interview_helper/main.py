"""Точка входа: python -m interview_helper [--config config.toml] [--file test.wav] [--list-devices]."""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import traceback

from .config import load_config
from .coordinator import INTERVIEWER, ME, Coordinator


def _dpi_aware() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="ИИ-помощник на собеседовании: Parakeet + Claude")
    ap.add_argument("--config", default="config.toml", help="путь к config.toml")
    ap.add_argument("--list-devices", action="store_true", help="показать звуковые устройства и выйти")
    ap.add_argument("--file", help="вместо звонка взять wav-файл как речь собеседника (для проверки)")
    ap.add_argument("--speed", type=float, default=1.0, help="скорость проигрывания --file")
    ap.add_argument("--no-llm", action="store_true", help="только расшифровка, без Claude")
    args = ap.parse_args()

    if args.list_devices:
        from .audio import list_devices
        list_devices()
        return

    _dpi_aware()
    cfg = load_config(args.config)
    ui_q: queue.Queue = queue.Queue()
    coord = Coordinator(cfg, ui_q)
    sources: list = []

    from .ui import OverlayUI

    def close() -> None:
        coord.stop()
        for s in sources:
            try:
                s.stop()
            except Exception:  # noqa: BLE001
                pass
        if coord.asr_worker:
            coord.asr_worker.stop()

    ui = OverlayUI(cfg.ui, ui_q, {
        "hint": lambda: coord.request_hint(),
        "ask": lambda text: coord.request_hint(text),
        "pause": coord.toggle_pause,
        "auto": coord.set_auto,
        "clear": coord.clear,
        "close": close,
    })
    ui.set_auto(cfg.llm.auto)
    coord.start()

    def boot() -> None:
        try:
            from .asr import AsrWorker, ParakeetASR
            from .audio import WasapiSource, WavFileSource
            from .segmenter import Segmenter
            from .vad import SileroVAD, ensure_vad_model

            ui_q.put(("loading", "загружаю VAD…"))
            vad_path = ensure_vad_model(cfg.path("models"))

            ui_q.put(("loading", "загружаю Parakeet (первый раз скачивается ~0,7–2,5 ГБ)…"))
            asr = ParakeetASR(cfg.asr)
            coord.asr_worker = AsrWorker(asr, coord.on_asr)

            if not args.no_llm:
                try:
                    from .llm import HintGenerator
                    coord.gen = HintGenerator(cfg)
                except Exception as e:  # noqa: BLE001
                    ui_q.put(("error", f"подсказки выключены: {e}"))

            v = cfg.vad
            seg_them = Segmenter(INTERVIEWER, SileroVAD(vad_path), v, coord.on_segment, partials=True)
            coord.segmenters.append(seg_them)
            if args.file:
                sources.append(WavFileSource(args.file, seg_them.feed, args.speed))
            else:
                if cfg.audio.capture_system:
                    sources.append(WasapiSource("system", seg_them.feed, cfg.audio.system_device))
                if cfg.audio.capture_mic:
                    seg_me = Segmenter(ME, SileroVAD(vad_path), v, coord.on_segment, partials=False)
                    coord.segmenters.append(seg_me)
                    sources.append(WasapiSource("mic", seg_me.feed, cfg.audio.mic_device))
            for s in sources:
                s.start()
            names = " + ".join(s.device_name for s in sources) or "нет источников звука"
            ui_q.put(("ready", "слушаю"))
            ui_q.put(("status", f"источник: {names}"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            ui_q.put(("loading", "ошибка запуска"))
            ui_q.put(("error", f"{type(e).__name__}: {e}"))

    threading.Thread(target=boot, name="boot", daemon=True).start()

    if cfg.ui.global_hotkeys:
        try:
            import keyboard
            keyboard.add_hotkey(cfg.ui.hotkey_hint, lambda: coord.request_hint())
            keyboard.add_hotkey(cfg.ui.hotkey_pause, coord.toggle_pause)
        except Exception as e:  # noqa: BLE001
            print(f"[hotkeys] глобальные горячие клавиши недоступны: {e}")

    ui.run()


if __name__ == "__main__":
    main()
