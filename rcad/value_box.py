"""On-model value box: shows the size being dragged and takes typed values.

Typed text accepts units (`2in`, `5cm`), arithmetic (`40/3 + 2`) and document
variables (`wall * 2`). Plain numbers are in the display unit; variables hold
millimetres, like sketch constraint values.
"""
from __future__ import annotations

import math
import re
from typing import Callable

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import QLineEdit

from .expression_eval import _evaluate_expression
from .units import Unit

UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, '"': 25.4, "ft": 304.8}

# A number directly followed by a unit, e.g. `2in`, `5 cm`, `1.5"`. The
# lookarounds keep `3min` or `x2m` from being read as a unit suffix.
_NUM_UNIT = re.compile(
    r'(?<![\w.])((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(mm|cm|m|in|ft|")(?![A-Za-z_\d])')


def parse_length(text: str, unit: Unit, variables: dict[str, float] | None = None) -> float:
    """Evaluate typed length text and return millimetres. Raises ValueError."""
    text = text.strip()
    if not text:
        raise ValueError("Empty value")
    # Work in display units: unit-suffixed numbers and variables (mm) are converted.
    expr = _NUM_UNIT.sub(lambda m: f"({m.group(1)}*{UNIT_MM[m.group(2)] / unit.mm!r})", text)
    names = {k: float(v) / unit.mm for k, v in (variables or {}).items()}
    try:
        value = _evaluate_expression(expr, names)
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError) as error:
        raise ValueError(f"Invalid value: {text}") from error
    if not math.isfinite(value):
        raise ValueError(f"Invalid value: {text}")
    return value * unit.mm


class ValueBox(QLineEdit):
    """Small editable label placed on the model next to a dragged size."""

    edited = Signal(float)    # live: text parses to this many mm
    accepted = Signal(float)  # Enter with a valid value (mm)
    cancelled = Signal()      # Esc

    COLORS = {"extrude": "0,128,0", "fillet": "128,0,128"}

    def __init__(self, parent, unit_fn: Callable[[], Unit], variables_fn: Callable[[], dict]):
        super().__init__(parent)
        self._unit_fn = unit_fn
        self._variables_fn = variables_fn
        self._kind = None
        self._valid = True
        self.typing = False  # the user owns the text; drags no longer overwrite it
        self.setFocusPolicy(Qt.ClickFocus)
        self.setAlignment(Qt.AlignCenter)
        self.textEdited.connect(self._on_text)
        self.returnPressed.connect(self._on_return)
        self._restyle()
        self.hide()

    # -- state
    def value(self) -> float | None:
        try:
            return parse_length(self.text(), self._unit_fn(), self._variables_fn())
        except ValueError:
            return None

    def show_value(self, mm: float):
        if not self.typing:
            self.setText(self._unit_fn().fmt(mm))
            self._fit()

    def start_typing(self, text: str):
        """Replace the shown value with the first typed character and take focus."""
        self.typing = True
        self.setText(text)
        self.show()
        self.raise_()
        self.setFocus()
        self._on_text(text)

    def end_typing(self):
        self.typing = False
        self._set_valid(True)
        self.clearFocus()

    def set_kind(self, kind: str):
        if kind != self._kind:
            self._kind = kind
            self._restyle()

    # -- events
    def _on_text(self, _text):
        self.typing = True
        v = self.value()
        self._set_valid(v is not None)
        self._fit()
        if v is not None:
            self.edited.emit(v)

    def _on_return(self):
        v = self.value()
        if v is None:
            self._set_valid(False)
            return
        self.accepted.emit(v)

    def event(self, e):
        # Keep Esc for the box while it has focus, ahead of the window's Esc shortcut.
        if e.type() == QEvent.ShortcutOverride and e.key() == Qt.Key_Escape:
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(e)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        if not self.typing:  # clicked: select the value so typing replaces it
            self.typing = True
            QTimer.singleShot(0, self.selectAll)

    # -- looks
    def _fit(self):
        w = self.fontMetrics().horizontalAdvance(self.text() or "0") + 18
        self.setFixedWidth(max(48, w))

    def _set_valid(self, valid: bool):
        if valid != self._valid:
            self._valid = valid
            self._restyle()

    def _restyle(self):
        rgb = self.COLORS.get(self._kind, "40,40,40")
        border = "#ff4040" if not self._valid else "transparent"
        self.setStyleSheet(
            f"QLineEdit {{ color: #ffffff; background: rgba({rgb},200); padding: 1px 4px; "
            f"border: 2px solid {border}; border-radius: 3px; font-weight: bold; }}")
