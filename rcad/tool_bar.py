"""Tool settings bar: the active tool's values and options along the bottom of the view.

Numbers can be typed (units, maths, variables) or dragged sideways; option words
cycle when clicked; ✓ (or Enter) finishes and ✕ (or Esc) cancels. The owner gets
``on_change(values)`` whenever every number is valid, so it can preview live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

from .units import Unit
from .value_box import ANGLE_STEP, SCRUB_START_PX, NumberField


@dataclass
class Number:
    key: str
    label: str
    value: float
    quantity: str = "length"  # "length" (mm) or "angle" (degrees)
    lo: float | None = None
    hi: float | None = None


@dataclass
class Choice:
    key: str
    label: str
    options: list[tuple[str, str]]  # (value, word shown)
    current: str


class ScrubField(NumberField):
    """NumberField that also changes by dragging sideways (Shift: fine, Ctrl: coarse)."""

    HINT = "Type a value, or drag sideways. " + NumberField.HINT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scrub = None

    def scrub_step(self, modifiers) -> float:
        step = ANGLE_STEP if self.quantity == "angle" else self.unit.snap / 10 * self.unit.mm
        if modifiers & Qt.ShiftModifier:
            step /= 10
        elif modifiers & Qt.ControlModifier:
            step *= 10
        return step

    def scrub_to(self, base: float, dx: float, modifiers=Qt.NoModifier) -> float:
        step = self.scrub_step(modifiers)
        v = round((base + dx * step) / step) * step
        if self.lo is not None:
            v = max(v, self.lo)
        if self.hi is not None:
            v = min(v, self.hi)
        self.setText(self.format(v))
        return v

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._scrub = {"x": e.globalPosition().x(), "base": self.value(), "active": False}
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        s = self._scrub
        if s is not None and s["base"] is not None and e.buttons() & Qt.LeftButton:
            dx = e.globalPosition().x() - s["x"]
            if s["active"] or abs(dx) > SCRUB_START_PX:
                s["active"] = True
                self.scrub_to(s["base"], dx, e.modifiers())
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        s, self._scrub = self._scrub, None
        if s is not None and s["active"]:
            self.selectAll()
            return
        super().mouseReleaseEvent(e)


class ToolSettingsBar(QFrame):
    """Bar of the active tool's settings, centred along the bottom of its parent (the view)."""

    active_changed = Signal(bool)  # shown for a tool (True) / put away (False)

    def __init__(self, parent: QWidget, unit_fn: Callable[[], Unit], variables_fn: Callable[[], dict]):
        super().__init__(parent)
        self._unit_fn = unit_fn
        self._variables_fn = variables_fn
        self.setObjectName("toolSettingsBar")
        self.setStyleSheet(
            "#toolSettingsBar { background: rgba(32,34,40,225); border-radius: 8px; }"
            "#toolSettingsBar QLabel { color: #e8e8ec; }"
            "#toolSettingsBar QLabel#title { font-weight: bold; }"
            "#toolSettingsBar QLabel#error { color: #ff7070; }"
            "#toolSettingsBar QPushButton { color: #ffffff; background: rgba(255,255,255,30); "
            "  border: 1px solid rgba(255,255,255,60); border-radius: 4px; padding: 2px 8px; }"
            "#toolSettingsBar QPushButton:hover { background: rgba(255,255,255,55); }"
            "#toolSettingsBar QToolButton { color: #ffffff; font-size: 15px; border: none; padding: 2px 6px; }"
            "#toolSettingsBar QToolButton:disabled { color: rgba(255,255,255,70); }"
            "#toolSettingsBar QToolButton:hover { background: rgba(255,255,255,40); border-radius: 4px; }")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(10, 6, 6, 6)
        self._layout.setSpacing(6)
        self.fields: dict[str, ScrubField] = {}
        self.choices: dict[str, tuple[QPushButton, Choice]] = {}
        self._on_change = self._on_finish = self._on_cancel = None
        self._error = None
        self.active = False
        parent.installEventFilter(self)
        self.hide()

    # -- session
    def begin(self, title: str, items: list, on_change: Callable[[dict], None],
              on_finish: Callable[[], None], on_cancel: Callable[[], None]):
        """Show the settings for a tool; replaces whatever the bar showed before."""
        self._clear()
        self._on_change, self._on_finish, self._on_cancel = on_change, on_finish, on_cancel
        lbl = QLabel(title, self)
        lbl.setObjectName("title")
        self._layout.addWidget(lbl)
        first = None
        for item in items:
            if isinstance(item, Number):
                self._layout.addWidget(QLabel(item.label, self))
                f = ScrubField(item.value, item.quantity, self._unit_fn(), self._variables_fn(),
                               item.lo, item.hi, parent=self)
                f.setFixedWidth(100)
                f.textChanged.connect(self._changed)
                f.validity_changed.connect(self._update_tick)
                f.returnPressed.connect(self.finish)
                self.fields[item.key] = f
                self._layout.addWidget(f)
                first = first or f
            else:
                self._layout.addWidget(QLabel(item.label, self))
                b = QPushButton(self)
                b.setToolTip("Click to switch: " + " / ".join(word for _v, word in item.options))
                b.clicked.connect(lambda _=False, k=item.key: self._cycle(k))
                self.choices[item.key] = (b, item)
                self._set_choice(item.key, item.current)
                self._layout.addWidget(b)
        self.error_label = QLabel(self)
        self.error_label.setObjectName("error")
        self.error_label.hide()
        self._layout.addWidget(self.error_label)
        self.tick = QToolButton(self)
        self.tick.setText("✓")
        self.tick.setToolTip("Done (Enter)")
        self.tick.clicked.connect(self.finish)
        self.cross = QToolButton(self)
        self.cross.setText("✕")
        self.cross.setToolTip("Cancel (Esc)")
        self.cross.clicked.connect(self.cancel)
        self._layout.addWidget(self.tick)
        self._layout.addWidget(self.cross)
        self.active = True
        self._error = None
        self.active_changed.emit(True)
        self._update_tick()
        self.show()
        self.raise_()
        self._place()
        if first is not None:
            first.setFocus()
            first.selectAll()

    def end(self):
        self._clear()
        was = self.active
        self.active = False
        self.hide()
        if was:
            self.active_changed.emit(False)

    def values(self) -> dict | None:
        """Every setting by key (numbers in mm or degrees), or None while a number is invalid."""
        out = {}
        for key, f in self.fields.items():
            v = f.value()
            if v is None:
                return None
            out[key] = v
        for key, (_b, item) in self.choices.items():
            out[key] = item.current
        return out

    def set_error(self, message: str | None):
        """Show why the current settings fail; ✓ stays off until they work."""
        self._error = message
        if not self.active:
            return
        self.error_label.setText(message or "")
        self.error_label.setVisible(bool(message))
        self._update_tick()
        self._place()

    def finish(self):
        if self.active and self.values() is not None and not self._error and self._on_finish:
            self._on_finish()

    def cancel(self):
        if self.active and self._on_cancel:
            self._on_cancel()

    # -- internals
    def _changed(self, *_):
        self._place()
        v = self.values()
        if v is not None and self._on_change:
            self._on_change(v)

    def _cycle(self, key: str):
        _b, item = self.choices[key]
        values = [v for v, _w in item.options]
        self._set_choice(key, values[(values.index(item.current) + 1) % len(values)])
        self._changed()

    def _set_choice(self, key: str, value: str):
        b, item = self.choices[key]
        item.current = value
        b.setText(dict(item.options)[value])

    def _update_tick(self, *_):
        if self.active:
            self.tick.setEnabled(self.values() is not None and not self._error)

    def _clear(self):
        while self._layout.count():
            w = self._layout.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self.fields, self.choices = {}, {}

    def _place(self):
        p = self.parentWidget()
        self.adjustSize()
        self.move(max((p.width() - self.width()) // 2, 0), max(p.height() - self.height() - 14, 0))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Resize and self.active:
            self._place()
        return False
