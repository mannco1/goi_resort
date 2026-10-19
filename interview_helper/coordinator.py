"""Мозг приложения: собирает реплики, решает, когда собеседник договорил, и запускает подсказку."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .asr import AsrResult
from .config import Config
from .segmenter import SegEvent, Segmenter

INTERVIEWER, ME = "interviewer", "me"
MERGE_GAP_S = 6.0   # фразы одного человека с паузой меньше этой склеиваются в одну реплику


@dataclass
class Turn:
    speaker: str
    text: str
    t_first: float
    t_last: float


class SessionLog:
    def __init__(self, cfg: Config):
        self.f = None
        if cfg.log.enabled:
            d = cfg.path(cfg.log.dir)
            d.mkdir(parents=True, exist_ok=True)
            p: Path = d / f"{datetime.now():%Y-%m-%d_%H-%M}.md"
            self.f = p.open("a", encoding="utf-8")
            self.f.write(f"# Собеседование {datetime.now():%d.%m.%Y %H:%M}\n\n")
            self.f.flush()

    def line(self, speaker: str, text: str, t: float) -> None:
        if self.f:
            who = "Интервьюер" if speaker == INTERVIEWER else "Я"
            self.f.write(f"**[{datetime.fromtimestamp(t):%H:%M:%S}] {who}:** {text}\n\n")
            self.f.flush()

    def hint(self, text: str) -> None:
        if self.f:
            quoted = "\n".join(f"> {ln}" for ln in text.strip().splitlines())
            self.f.write(f"{quoted}\n\n")
            self.f.flush()


class Coordinator:
    def __init__(self, cfg: Config, ui_q: queue.Queue, hint_gen=None):
        self.cfg = cfg
        self.ui_q = ui_q
        self.gen = hint_gen
        self.inbox: queue.Queue = queue.Queue()
        self.asr_worker = None
        self.segmenters: list[Segmenter] = []
        self.log = SessionLog(cfg)

        self.turns: list[Turn] = []
        self.live: dict[str, tuple[int, str]] = {}
        self.last_final: dict[str, int] = {INTERVIEWER: 0, ME: 0}
        self.speaking: dict[str, int] = {}          # source -> seg_id текущей незакрытой фразы
        self.trigger_at: float | None = None
        self.answered_marker: tuple[int, int] | None = None
        self.sealed_idx = -1                        # реплика, на которую уже дали подсказку — к ней не приклеиваем
        self.auto = cfg.llm.auto
        self.paused = False

        self.hint_id = 0
        self.hint_cancel = threading.Event()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="coordinator", daemon=True)

    # ---- вход из других потоков (все потокобезопасны) ----
    def on_segment(self, ev: SegEvent) -> None:
        if self.asr_worker is not None and ev.kind in ("partial", "final"):
            self.asr_worker.submit(ev)
        self.inbox.put(("seg", ev))

    def on_asr(self, res: AsrResult) -> None:
        self.inbox.put(("asr", res))

    def request_hint(self, question: str | None = None) -> None:
        self.inbox.put(("hint", question))

    def toggle_pause(self) -> None:
        self.inbox.put(("pause", None))

    def set_auto(self, value: bool) -> None:
        self.inbox.put(("auto", value))

    def clear(self) -> None:
        self.inbox.put(("clear", None))

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.hint_cancel.set()

    # ---- основной цикл ----
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                kind, payload = self.inbox.get(timeout=0.1)
                getattr(self, f"_on_{kind}")(payload)
            except queue.Empty:
                pass
            except Exception as e:  # noqa: BLE001
                self.ui_q.put(("error", f"{type(e).__name__}: {e}"))
            if self.trigger_at is not None and time.time() >= self.trigger_at:
                self.trigger_at = None
                if self.auto and not self.paused:
                    self._auto_hint()

    def _on_seg(self, ev: SegEvent) -> None:
        if ev.kind == "start":
            self.speaking[ev.source] = ev.seg_id
            if ev.source == INTERVIEWER:
                self.trigger_at = None              # собеседник продолжает говорить — ждём
            self.live[ev.source] = (ev.seg_id, "…")
        elif ev.kind == "discard":
            if self.speaking.get(ev.source) == ev.seg_id:
                del self.speaking[ev.source]
            self._close_live(ev.source, ev.seg_id)
            if ev.source == INTERVIEWER and self._unanswered():
                self._arm_trigger()
        elif ev.kind == "final":
            self.speaking.pop(ev.source, None)
            return                                  # текст придёт от ASR
        else:
            return
        self._push_transcript()

    def _on_asr(self, r: AsrResult) -> None:
        if r.kind == "partial":
            if r.seg_id > self.last_final.get(r.source, 0) and r.text:
                self.live[r.source] = (r.seg_id, r.text)
                self._push_transcript()
            return

        self.last_final[r.source] = max(self.last_final.get(r.source, 0), r.seg_id)
        self._close_live(r.source, r.seg_id)
        if r.text:
            last = self.turns[-1] if self.turns else None
            if (last and last.speaker == r.source and r.t - last.t_last < MERGE_GAP_S
                    and len(self.turns) - 1 != self.sealed_idx):
                last.text = f"{last.text} {r.text}"
                last.t_last = r.t
            else:
                self.turns.append(Turn(r.source, r.text, r.t, r.t))
            self.log.line(r.source, r.text, r.t)
            self.ui_q.put(("status", f"распознано за {r.took:.2f} с"))
        if r.source == INTERVIEWER and INTERVIEWER not in self.speaking and self._unanswered():
            self._arm_trigger()
        self._push_transcript()

    def _on_hint(self, question: str | None) -> None:
        if question:
            self._start_hint(question.strip(), exclude_idx=None)
            return
        q = self._last_interviewer_text(include_live=True)
        if not q:
            self.ui_q.put(("status", "пока нечего подсказывать — собеседник ничего не сказал"))
            return
        i = self._last_interviewer_idx()
        if i is not None:
            self.answered_marker = (i, len(self.turns[i].text))
            self.sealed_idx = i
        self._start_hint(q, exclude_idx=i)

    def _on_pause(self, _) -> None:
        self.paused = not self.paused
        for s in self.segmenters:
            s.set_paused(self.paused)
        self.trigger_at = None
        self.ui_q.put(("paused", self.paused))

    def _on_auto(self, value: bool) -> None:
        self.auto = value
        if not value:
            self.trigger_at = None

    def _on_clear(self, _) -> None:
        self.hint_cancel.set()
        self.turns.clear()
        self.live.clear()
        self.answered_marker = None
        self.sealed_idx = -1
        self.trigger_at = None
        self._push_transcript()
        self.ui_q.put(("hint_clear", None))

    # ---- логика подсказок ----
    def _arm_trigger(self) -> None:
        self.trigger_at = time.time() + self.cfg.llm.turn_pause_ms / 1000

    def _last_interviewer_idx(self) -> int | None:
        for i in range(len(self.turns) - 1, -1, -1):
            if self.turns[i].speaker == INTERVIEWER:
                return i
        return None

    def _unanswered(self) -> bool:
        i = self._last_interviewer_idx()
        return i is not None and self.answered_marker != (i, len(self.turns[i].text))

    def _last_interviewer_text(self, include_live: bool) -> str:
        i = self._last_interviewer_idx()
        text = self.turns[i].text if i is not None else ""
        if include_live and INTERVIEWER in self.live:
            live = self.live[INTERVIEWER][1]
            if live and live != "…":
                text = f"{text} {live}".strip()
        return text

    def _auto_hint(self) -> None:
        i = self._last_interviewer_idx()
        if i is None or i != len(self.turns) - 1:
            return                                  # последним говорили вы — значит, вы уже отвечаете
        turn = self.turns[i]
        if len(turn.text.split()) < self.cfg.llm.min_words or not self._unanswered():
            return
        self.answered_marker = (i, len(turn.text))
        self.sealed_idx = i
        self._start_hint(turn.text, exclude_idx=i)

    def _start_hint(self, question: str, exclude_idx: int | None) -> None:
        if self.gen is None:
            self.ui_q.put(("status", "Claude API не настроен — работает только расшифровка"))
            return
        self.hint_cancel.set()                      # старая подсказка больше не нужна
        cancel = threading.Event()
        self.hint_cancel = cancel
        self.hint_id += 1
        hid = self.hint_id
        history = [(t.speaker, t.text) for k, t in enumerate(self.turns) if k != exclude_idx]
        history = history[-self.cfg.llm.history_turns:]
        self.ui_q.put(("hint_start", (hid, question, exclude_idx is None)))

        def work() -> None:
            t0 = time.perf_counter()
            try:
                full = self.gen.stream(history, question,
                                       lambda d: self.ui_q.put(("hint_delta", (hid, d))), cancel)
            except Exception as e:  # noqa: BLE001
                if not cancel.is_set():
                    self.ui_q.put(("error", f"Claude API: {e}"))
                return
            if not cancel.is_set():
                self.ui_q.put(("hint_done", (hid, full, time.perf_counter() - t0)))
                if full.strip() not in ("", "—"):
                    self.log.hint(full)

        threading.Thread(target=work, name=f"hint-{hid}", daemon=True).start()

    # ---- вывод ----
    def _close_live(self, source: str, seg_id: int) -> None:
        cur = self.live.get(source)
        if cur and cur[0] <= seg_id:
            del self.live[source]

    def _push_transcript(self) -> None:
        turns = [(t.speaker, t.text) for t in self.turns[-40:]]
        live = [(src, text) for src, (_, text) in self.live.items()]
        self.ui_q.put(("transcript", (turns, live)))
