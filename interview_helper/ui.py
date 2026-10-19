"""Окно-оверлей поверх всех окон (tkinter, без лишних зависимостей)."""
from __future__ import annotations

import queue
import re
import sys
import tkinter as tk
from tkinter import font as tkfont

from .config import UiCfg

C = {
    "bg": "#14161b", "panel": "#1b1e25", "line": "#2a2e38",
    "fg": "#e9e7e3", "muted": "#8a909b", "dim": "#5d636e",
    "them": "#79c9a6", "me": "#8fb0ff", "bold": "#f2bd72",
    "ok": "#79c9a6", "warn": "#f2bd72", "err": "#ef7f7f",
}
BOLD_RE = re.compile(r"(\*\*[^*]+\*\*)")


class OverlayUI:
    def __init__(self, cfg: UiCfg, ui_q: queue.Queue, actions: dict):
        self.cfg = cfg
        self.q = ui_q
        self.actions = actions  # hint, pause, auto(bool), clear, ask(str), close
        self.root = tk.Tk()
        self.root.title("Interview Helper")
        self.root.configure(bg=C["bg"])
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", cfg.opacity)
        sw = self.root.winfo_screenwidth()
        x = cfg.x if cfg.x >= 0 else sw - cfg.width - 24
        self.root.geometry(f"{cfg.width}x{cfg.height}+{x}+{cfg.y}")
        self.root.minsize(340, 300)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        family = "Segoe UI" if sys.platform == "win32" else tkfont.nametofont("TkDefaultFont").actual("family")
        fs = cfg.font_size
        self.f_hint = (family, fs + 1)
        self.f_hint_b = (family, fs + 1, "bold")
        self.f_text = (family, fs - 1)
        self.f_small = (family, fs - 3)
        self.f_btn = (family, fs - 2)

        self._build()
        self.hid = 0
        self.hint_buf = ""
        self.hint_shown = False
        self.prev_hint: tuple[str, str] | None = None   # (вопрос, текст) — вернуть, если пришло «—»
        self.cur_question = ""
        self._bind_keys()
        self.root.after(50, self._poll)
        if cfg.hide_from_capture:
            self.root.after(300, self._hide_from_capture)

    # ---------- построение ----------
    def _btn(self, parent, text, cmd):
        b = tk.Button(parent, text=text, command=cmd, font=self.f_btn, bg=C["panel"], fg=C["fg"],
                      activebackground=C["line"], activeforeground=C["fg"], relief="flat",
                      bd=0, padx=10, pady=4, cursor="hand2", highlightthickness=0)
        b.pack(side="left", padx=(0, 6))
        return b

    def _build(self) -> None:
        top = tk.Frame(self.root, bg=C["bg"])
        top.pack(fill="x", padx=12, pady=(10, 6))
        self.dot = tk.Label(top, text="●", fg=C["warn"], bg=C["bg"], font=self.f_btn)
        self.dot.pack(side="left")
        self.state_lbl = tk.Label(top, text="запуск…", fg=C["muted"], bg=C["bg"], font=self.f_btn)
        self.state_lbl.pack(side="left", padx=(4, 12))

        bar = tk.Frame(top, bg=C["bg"])
        bar.pack(side="right")
        self._btn(bar, "Ответ  F2", lambda: self.actions["hint"]())
        self.pause_btn = self._btn(bar, "Пауза  F3", lambda: self.actions["pause"]())
        self.auto_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Авто", variable=self.auto_var, command=self._toggle_auto,
                       font=self.f_btn, bg=C["bg"], fg=C["fg"], selectcolor=C["panel"],
                       activebackground=C["bg"], activeforeground=C["fg"], bd=0,
                       highlightthickness=0).pack(side="left", padx=(0, 6))
        self._btn(bar, "⌫", lambda: self.actions["clear"]())

        bottom = tk.Frame(self.root, bg=C["bg"])
        bottom.pack(side="bottom", fill="x", padx=12, pady=(6, 10))
        self.entry = tk.Entry(bottom, font=self.f_text, bg=C["panel"], fg=C["fg"], insertbackground=C["fg"],
                              relief="flat", highlightthickness=1, highlightbackground=C["line"],
                              highlightcolor=C["dim"])
        self.entry.pack(fill="x", ipady=5)
        self.entry.bind("<Return>", self._ask_typed)
        self.status = tk.Label(bottom, text="свой вопрос: напишите и нажмите Enter", fg=C["dim"],
                               bg=C["bg"], font=self.f_small, anchor="w")
        self.status.pack(fill="x", pady=(4, 0))

        panes = tk.PanedWindow(self.root, orient="vertical", bg=C["bg"], sashwidth=6, bd=0, sashrelief="flat")
        panes.pack(fill="both", expand=True, padx=12)

        hint_box = tk.Frame(panes, bg=C["panel"])
        self.q_lbl = tk.Label(hint_box, text="Подсказка появится, когда собеседник задаст вопрос",
                              fg=C["muted"], bg=C["panel"], font=self.f_small, anchor="w",
                              justify="left", wraplength=self.cfg.width - 60)
        self.q_lbl.pack(fill="x", padx=12, pady=(10, 2))
        self.hint = self._text(hint_box, self.f_hint)
        self.hint.tag_configure("b", font=self.f_hint_b, foreground=C["bold"])
        self.hint.tag_configure("muted", foreground=C["dim"])
        self.hint.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        panes.add(hint_box, stretch="always", minsize=160)

        tr_box = tk.Frame(panes, bg=C["bg"])
        self.tr = self._text(tr_box, self.f_text, bg=C["bg"])
        self.tr.tag_configure("them", foreground=C["them"], font=(self.f_text[0], self.f_text[1], "bold"))
        self.tr.tag_configure("me", foreground=C["me"], font=(self.f_text[0], self.f_text[1], "bold"))
        self.tr.tag_configure("live", foreground=C["muted"], font=(self.f_text[0], self.f_text[1], "italic"))
        self.tr.pack(fill="both", expand=True, pady=(6, 0))
        panes.add(tr_box, stretch="always", minsize=90)
        self.panes = panes
        self.root.after(120, lambda: panes.sash_place(0, 0, int(panes.winfo_height() * 0.64)))

    def _text(self, parent, fnt, bg=None) -> tk.Text:
        t = tk.Text(parent, wrap="word", font=fnt, bg=bg or C["panel"], fg=C["fg"], relief="flat", bd=0,
                    padx=8, pady=4, highlightthickness=0, cursor="arrow", spacing1=2, spacing3=4)
        t.configure(state="disabled")
        return t

    def _bind_keys(self) -> None:
        self.root.bind("<F2>", lambda e: self.actions["hint"]())
        self.root.bind("<F3>", lambda e: self.actions["pause"]())
        self.root.bind("<Escape>", lambda e: self.root.focus_set())

    # ---------- действия ----------
    def _toggle_auto(self) -> None:
        self.actions["auto"](self.auto_var.get())

    def _ask_typed(self, _e=None) -> None:
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self.actions["ask"](text)

    def _close(self) -> None:
        try:
            self.actions["close"]()
        finally:
            self.root.destroy()

    def _hide_from_capture(self) -> None:
        if sys.platform != "win32":
            return
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        WDA_EXCLUDEFROMCAPTURE = 0x11
        if not ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            self._set_status("не удалось скрыть окно от записи экрана (нужна Windows 10 2004+)", C["warn"])

    def set_auto(self, value: bool) -> None:
        self.auto_var.set(value)

    # ---------- отрисовка ----------
    def _set_state(self, text: str, color: str) -> None:
        self.state_lbl.configure(text=text)
        self.dot.configure(fg=color)

    def _set_status(self, text: str, color: str | None = None) -> None:
        self.status.configure(text=text, fg=color or C["dim"])

    def _render_hint(self, text: str) -> None:
        h = self.hint
        h.configure(state="normal")
        h.delete("1.0", "end")
        for line in text.strip("\n").splitlines():
            s = line.rstrip()
            if s.lstrip().startswith(("- ", "* ", "• ")):
                s = "•  " + s.lstrip()[2:].lstrip()
            for part in BOLD_RE.split(s):
                if part.startswith("**") and part.endswith("**") and len(part) > 4:
                    h.insert("end", part[2:-2], "b")
                elif part:
                    h.insert("end", part)
            h.insert("end", "\n")
        h.configure(state="disabled")

    def _render_transcript(self, turns, live) -> None:
        t = self.tr
        t.configure(state="normal")
        t.delete("1.0", "end")
        for who, text in turns:
            tag, name = ("them", "Он(а)") if who == "interviewer" else ("me", "Я")
            t.insert("end", f"{name}: ", tag)
            t.insert("end", f"{text}\n")
        for who, text in live:
            tag, name = ("them", "Он(а)") if who == "interviewer" else ("me", "Я")
            t.insert("end", f"{name}: ", tag)
            t.insert("end", f"{text}\n", "live")
        t.configure(state="disabled")
        t.see("end")

    def _poll(self) -> None:
        try:
            for _ in range(200):
                kind, data = self.q.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        self.root.after(40, self._poll)

    def _handle(self, kind: str, data) -> None:
        if kind == "transcript":
            self._render_transcript(*data)
        elif kind == "ready":
            self._set_state(data, C["ok"])
        elif kind == "loading":
            self._set_state(data, C["warn"])
        elif kind == "paused":
            if data:
                self._set_state("пауза — не слушаю", C["warn"])
                self.pause_btn.configure(text="Слушать  F3")
            else:
                self._set_state("слушаю", C["ok"])
                self.pause_btn.configure(text="Пауза  F3")
        elif kind == "status":
            self._set_status(data)
        elif kind == "error":
            self._set_status(data, C["err"])
        elif kind == "hint_start":
            hid, question, manual = data
            self.hid, self.hint_buf, self.hint_shown = hid, "", False
            self.cur_question = question
            short = question if len(question) <= 220 else "…" + question[-220:]
            self.q_lbl.configure(text=f"⟳  {short}", fg=C["muted"])
        elif kind == "hint_delta":
            hid, delta = data
            if hid != self.hid:
                return
            self.hint_buf += delta
            if self.hint_shown or self.hint_buf.strip() not in ("", "—"):
                self.hint_shown = True
                self._render_hint(self.hint_buf)
        elif kind == "hint_done":
            hid, full, took = data
            if hid != self.hid:
                return
            if full.strip() in ("", "—"):
                # не вопрос — оставляем предыдущую подсказку на экране
                if self.prev_hint:
                    q, txt = self.prev_hint
                    self.q_lbl.configure(text=f"❝  {q}", fg=C["muted"])
                    self._render_hint(txt)
                else:
                    self.q_lbl.configure(text="Подсказка появится, когда собеседник задаст вопрос", fg=C["muted"])
                self._set_status("последняя реплика — не вопрос, подсказка не нужна")
                return
            short = self.cur_question if len(self.cur_question) <= 220 else "…" + self.cur_question[-220:]
            self.q_lbl.configure(text=f"❝  {short}", fg=C["muted"])
            self.prev_hint = (short, full)
            self._set_status(f"подсказка за {took:.1f} с")
        elif kind == "hint_clear":
            self.prev_hint = None
            self.hid += 1
            self._render_hint("")
            self.q_lbl.configure(text="Подсказка появится, когда собеседник задаст вопрос", fg=C["muted"])

    def run(self) -> None:
        self.root.mainloop()
