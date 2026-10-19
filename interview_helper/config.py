"""Загрузка настроек из config.toml поверх значений по умолчанию."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class AudioCfg:
    capture_system: bool = True      # звук собеседника (то, что играет в наушниках/колонках)
    capture_mic: bool = True         # ваш микрофон (для контекста, подсказки не запускает)
    system_device: str = ""          # часть имени устройства вывода; пусто = устройство по умолчанию
    mic_device: str = ""             # часть имени микрофона; пусто = по умолчанию


@dataclass
class VadCfg:
    threshold: float = 0.5           # порог вероятности речи Silero
    silence_ms: int = 600            # сколько тишины закрывает фразу
    min_speech_ms: int = 300         # короче — считаем шумом
    max_segment_s: float = 20.0      # принудительно режем длинные монологи
    partial_interval_s: float = 1.2  # как часто обновлять «живой» текст собеседника
    preroll_ms: int = 250            # сколько звука до начала речи захватываем


@dataclass
class AsrCfg:
    model: str = "nemo-parakeet-tdt-0.6b-v3"
    quantization: str = "int8"       # "int8" быстрее и легче; "" — полная точность
    providers: list[str] = field(default_factory=list)  # например ["DmlExecutionProvider"] или ["CUDAExecutionProvider"]
    model_dir: str = ""              # путь к локальной папке с ONNX-моделью (если скачали вручную)


@dataclass
class LlmCfg:
    model: str = "claude-haiku-5-5"  # самый быстрый; для глубоких ответов — "claude-sonnet-5-5"
    api_key: str = ""                # если пусто — берётся из переменной ANTHROPIC_API_KEY
    max_tokens: int = 450
    auto: bool = True                # подсказка сама появляется, когда собеседник договорил
    turn_pause_ms: int = 900         # пауза после фразы собеседника, после которой считаем, что он договорил
    min_words: int = 4               # короткие реплики («угу», «понятно») не триггерят подсказку
    history_turns: int = 14          # сколько последних реплик отдавать модели как контекст


@dataclass
class ProfileCfg:
    resume: str = "profile/resume.md"
    vacancy: str = "profile/vacancy.md"
    notes: str = "profile/notes.md"


@dataclass
class UiCfg:
    opacity: float = 0.93
    font_size: int = 13
    width: int = 520
    height: int = 640
    x: int = -1                      # -1 = у правого края экрана
    y: int = 60
    hide_from_capture: bool = False  # не показывать окно в демонстрации экрана/записи (Windows 10 2004+)
    global_hotkeys: bool = True
    hotkey_hint: str = "ctrl+alt+space"
    hotkey_pause: str = "ctrl+alt+p"


@dataclass
class LogCfg:
    enabled: bool = True
    dir: str = "sessions"


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    profile: ProfileCfg = field(default_factory=ProfileCfg)
    ui: UiCfg = field(default_factory=UiCfg)
    log: LogCfg = field(default_factory=LogCfg)
    base_dir: Path = field(default_factory=Path.cwd)

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.base_dir / q

    @property
    def api_key(self) -> str:
        return self.llm.api_key or os.environ.get("ANTHROPIC_API_KEY", "")


def _merge(obj, data: dict, section: str = "") -> None:
    names = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            print(f"[config] неизвестный параметр {section}{key} — пропускаю")
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{section}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path:
        p = Path(path).resolve()
        cfg.base_dir = p.parent
        if p.exists():
            with p.open("rb") as f:
                _merge(cfg, tomllib.load(f))
        else:
            print(f"[config] {p} не найден — работаю на настройках по умолчанию")
    return cfg
