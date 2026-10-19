"""Генерация подсказок через Claude API (стриминг, с отменой)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from .config import Config

SYSTEM_PROMPT = """\
Ты — суфлёр кандидата на собеседовании в реальном времени. Тебе приходит расшифровка разговора \
из системы распознавания речи — в ней бывают ошибки и обрывки, восстанавливай смысл по контексту. \
Задача: за секунды дать кандидату опору для ответа на последний вопрос интервьюера.

Правила:
- Отвечай на языке вопроса.
- Пиши от первого лица кандидата и опирайся только на факты из резюме и заметок ниже. \
Не выдумывай опыт, компании, цифры и инструменты. Если подходящего опыта нет — предложи честную \
формулировку: как связать смежный опыт или как ответить, что напрямую не делал, но как бы подошёл.
- Формат строго такой:
**Суть:** что на самом деле хотят узнать (одна строка)
- 3–5 коротких тезисов, которые можно сказать вслух (каждый не длиннее 15 слов)
**Пример:** один конкретный случай из опыта в 1–2 предложениях (если уместно)
**Уточнить:** встречный вопрос интервьюеру (только если уместно)
- Технический вопрос: сначала верный короткий ответ по сути, затем как применял это на практике.
- Кейс или задача: шаги решения по порядку.
- Вопрос про зарплату, уход с работы, слабые стороны: дипломатичная формулировка без лишних деталей.
- Если последняя реплика — не вопрос и не просьба (рассказ о компании, small talk, «угу», «отлично»), \
ответь ровно одним символом: —
- Никаких вступлений, извинений и комментариев от себя.
"""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


class HintGenerator:
    def __init__(self, cfg: Config):
        import anthropic

        if not cfg.api_key:
            raise RuntimeError("Нет ключа Claude API: задайте ANTHROPIC_API_KEY или llm.api_key в config.toml")
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=cfg.api_key)
        self.reload_profile()

    def reload_profile(self) -> None:
        p = self.cfg.profile
        parts = [SYSTEM_PROMPT]
        for title, path in (("РЕЗЮМЕ КАНДИДАТА", p.resume), ("ВАКАНСИЯ", p.vacancy), ("ЗАМЕТКИ КАНДИДАТА", p.notes)):
            text = _read(self.cfg.path(path))
            if text:
                parts.append(f"<{title}>\n{text}\n</{title}>")
        # большой неизменный блок кешируется на стороне API — повторные подсказки приходят быстрее и дешевле
        self.system = [{"type": "text", "text": "\n\n".join(parts), "cache_control": {"type": "ephemeral"}}]

    def build_user_message(self, history: list[tuple[str, str]], question: str) -> str:
        lines = [f"{'Интервьюер' if who == 'interviewer' else 'Я'}: {text}" for who, text in history]
        convo = "\n".join(lines) if lines else "(начало разговора)"
        return (f"Расшифровка разговора (последние реплики):\n{convo}\n\n"
                f"Последняя реплика интервьюера, на которую нужна подсказка:\n«{question}»")

    def stream(self, history: list[tuple[str, str]], question: str,
               on_delta: Callable[[str], None], cancel: threading.Event) -> str:
        """Стримит ответ; возвращает полный текст. Прерывается, если выставлен cancel."""
        out: list[str] = []
        with self.client.messages.stream(
            model=self.cfg.llm.model,
            max_tokens=self.cfg.llm.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": self.build_user_message(history, question)}],
        ) as s:
            for chunk in s.text_stream:
                if cancel.is_set():
                    break
                out.append(chunk)
                on_delta(chunk)
        return "".join(out)
