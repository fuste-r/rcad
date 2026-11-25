"""On-model value boxes: show the size being dragged or drawn and take typed values.

Typed text accepts units (`2in`, `5cm`), arithmetic (`40/3 + 2`) and document
variables (`wall * 2`). Plain numbers are in the display unit; variables hold
millimetres, like sketch constraint values. Angles are in degrees.

Dragging a box left or right scrubs its value (Shift: fine, Ctrl: coarse).
"""
from __future__ import annotations

import math
import re
from typing import Callable

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from .expression_eval import _evaluate_expression
from .units import Unit

UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, '"': 25.4, "ft": 304.8}

# A number directly followed by a unit, e.g. `2in`, `5 cm`, `1.5"`. The
# lookarounds keep `3min` or `x2m` from being read as a unit suffix.
_NUM = r'(?<![\w.])((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
_NUM_UNIT = re.compile(_NUM + r'\s*(mm|cm|m|in|ft|")(?![A-Za-z_\d])')
_NUM_DEG = re.compile(_NUM + r'\s*(?:°|deg)(?![A-Za-z_\d])')

SCRUB_START_PX = 3
ANGLE_STEP = 0.5  # degrees per pixel of scrub


def _evaluate(text: str, expr: str, names: dict[str, float]) -> float:
    try:
        value = _evaluate_expression(expr, names)
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError) as error:
        raise ValueError(f"Invalid value: {text}") from error
    if not math.isfinite(value):
        raise ValueError(f"Invalid value: {text}")
    return value


def parse_length(text: str, unit: Unit, variables: dict[str, float] | None = None) -> float:
    """Evaluate typed length text and return millimetres. Raises ValueError."""
    text = text.strip()
    if not text:
        raise ValueError("Empty value")
    # Work in display units: unit-suffixed numbers and variables (mm) are converted.
    expr = _NUM_UNIT.sub(lambda m: f"({m.group(1)}*{UNIT_MM[m.group(2)] / unit.mm!r})", text)
    names = {k: float(v) / unit.mm for k, v in (variables or {}).items()}
    return _evaluate(text, expr, names) * unit.mm


def parse_angle(text: str, variables: dict[str, float] | None = None) -> float:
    """Evaluate typed angle text (degrees; `°` or `deg` allowed) and return degrees."""
    text = text.strip()
    if not text:
        raise ValueError("Empty value")
    expr = _NUM_DEG.sub(lambda m: m.group(1), text)
    return _evaluate(text, expr, {k: float(v) for k, v in (variables or {}).items()})


def format_angle(deg: float) -> str:
    return f"{round(deg, 2):g}°"


class ValueBox(QLineEdit):
    """Small editable value placed on the model next to a dragged or drawn size."""

    edited = Signal(float)          # live: typed text parses to this value (mm or degrees)
    accepted = Signal(float)        # Enter with a valid value
    cancelled = Signal()            # Esc
    tab = Signal(bool)              # Tab (True) / Shift+Tab (False); focus never leaves the box
    scrubbed = Signal(float)        # live value while dragging the box sideways
    scrub_finished = Signal(float)  # mouse released after a scrub
    focused = Signal()

    COLORS = {"extrude": "0,128,0", "fillet": "128,0,128", "sketch": "30,90,200"}

    def __init__(self, parent, unit_fn: Callable[[], Unit], variables_fn: Callable[[], dict],
                 quantity: str = "length"):
        super().__init__(parent)
        self._unit_fn = unit_fn
        self._variables_fn = variables_fn
        self.quantity = quantity  # "length" (mm) or "angle" (degrees)
        self._kind = None
        self._valid = True
        self._lit = None  # None: lone box; True/False: lit or dimmed member of a FieldSet
        self._scrub = None  # {"x", "base", "active"} while the left button is down on the box
        self.typing = False  # the user set the text (typed or scrubbed); updates no longer overwrite it
        self.setFocusPolicy(Qt.ClickFocus)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip("Type a value, or drag sideways to change it (Shift: fine, Ctrl: coarse)")
        self.textEdited.connect(self._on_text)
        self.returnPressed.connect(self._on_return)
        self._restyle()
        self.hide()

    # -- values
    def parse(self, text: str) -> float:
        if self.quantity == "angle":
            return parse_angle(text, self._variables_fn())
        return parse_length(text, self._unit_fn(), self._variables_fn())

    def format(self, value: float) -> str:
        return format_angle(value) if self.quantity == "angle" else self._unit_fn().fmt(value)

    def value(self) -> float | None:
        try:
            return self.parse(self.text())
        except ValueError:
            return None

    def show_value(self, value: float):
        """Display a value that comes from the mouse, unless the user owns the text."""
        if self.typing:
            return
        text = self.format(value)
        if text != self.text():
            self.setText(text)
            self._fit()
            if self.hasFocus():
                self.selectAll()  # typing still replaces the shown value

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
        self._scrub = None
        self._set_valid(True)
        self.clearFocus()

    def set_kind(self, kind: str):
        if kind != self._kind:
            self._kind = kind
            self._restyle()

    def set_lit(self, lit: bool | None):
        if lit != self._lit:
            self._lit = lit
            self._restyle()

    # -- typing
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

    def focusNextPrevChild(self, forward: bool) -> bool:
        self.tab.emit(forward)
        return True  # handled: focus stays on the value boxes

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(e)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        self.setCursor(Qt.IBeamCursor)
        self.focused.emit()
        if not self.typing:  # clicked or tabbed to: select the value so typing replaces it
            QTimer.singleShot(0, self.selectAll)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.setCursor(Qt.SizeHorCursor)

    # -- scrubbing
    def scrub_step(self, modifiers) -> float:
        """Value change per pixel of horizontal drag, in mm or degrees."""
        step = ANGLE_STEP if self.quantity == "angle" else self._unit_fn().snap / 10 * self._unit_fn().mm
        if modifiers & Qt.ShiftModifier:
            step /= 10
        elif modifiers & Qt.ControlModifier:
            step *= 10
        return step

    def scrub_to(self, dx: float, modifiers=Qt.NoModifier) -> float | None:
        """Set the value to the scrub base moved by dx pixels; returns it (None if no base)."""
        s = self._scrub
        if s is None or s["base"] is None:
            return None
        step = self.scrub_step(modifiers)
        v = round((s["base"] + dx * step) / step) * step
        self.typing = True
        self.setText(self.format(v))
        self._set_valid(True)
        self._fit()
        self.scrubbed.emit(v)
        return v

    def begin_scrub(self, x: float):
        self._scrub = {"x": x, "base": self.value(), "active": False, "last": None}

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.begin_scrub(e.globalPosition().x())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        s = self._scrub
        if s is not None and s["base"] is not None and e.buttons() & Qt.LeftButton:
            # Global coordinates: the box itself moves when the value moves its anchor.
            dx = e.globalPosition().x() - s["x"]
            if s["active"] or abs(dx) > SCRUB_START_PX:
                s["active"] = True
                s["last"] = self.scrub_to(dx, e.modifiers())
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        s, self._scrub = self._scrub, None
        if s is not None and s["active"]:
            self.selectAll()
            if s["last"] is not None:
                self.scrub_finished.emit(s["last"])
            return
        super().mouseReleaseEvent(e)

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
        alpha = 120 if self._lit is False else 200
        border = "#ff4040" if not self._valid else ("#ffffff" if self._lit else "transparent")
        self.setStyleSheet(
            f"QLineEdit {{ color: #ffffff; background: rgba({rgb},{alpha}); padding: 1px 4px; "
            f"border: 2px solid {border}; border-radius: 3px; font-weight: bold; }}")


class FieldSet(QObject):
    """A row of value boxes by the pointer for a tool with several sizes.

    The mouse sets every size the user hasn't typed; a typed (or scrubbed) size
    stays fixed. Tab and Shift+Tab move between the boxes, Enter places the shape,
    Esc cancels. Box values are mm (lengths) or degrees (angles).

    With ``post`` the boxes belong to a shape already placed: they stay on it
    (at ``anchor``, a world point the view keeps them next to) and edit its sizes.
    """

    changed = Signal()    # a fixed size changed: refresh the preview
    accepted = Signal()   # Enter: place the shape
    cancelled = Signal()  # Esc

    def __init__(self, parent: QWidget, unit_fn: Callable[[], Unit], variables_fn: Callable[[], dict]):
        super().__init__(parent)
        self._parent = parent
        self._unit_fn = unit_fn
        self._variables_fn = variables_fn
        self.row = QWidget(parent)
        self._layout = QHBoxLayout(self.row)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(3)
        self.row.hide()
        self.keys: list[str] = []
        self.boxes: dict[str, ValueBox] = {}
        self._labels: list[QLabel] = []
        self.active_index = 0
        self.active = False
        self.post = False
        self.anchor = None  # world point the boxes sit next to in post mode

    def start(self, specs: list[tuple[str, str, str]], post: bool = False):
        """Show boxes for (key, label, quantity) specs with nothing fixed yet."""
        if [s[0] for s in specs] != self.keys:
            self._clear()
            for key, label, quantity in specs:
                lbl = QLabel(label, self.row)
                lbl.setStyleSheet("color: #ffffff; background: rgba(30,90,200,200); padding: 1px 4px; "
                                  "border-radius: 3px; font-weight: bold;")
                box = ValueBox(self.row, self._unit_fn, self._variables_fn, quantity)
                box.set_kind("sketch")
                box.show()
                box.edited.connect(lambda _v: self.changed.emit())
                box.scrubbed.connect(lambda _v: self.changed.emit())
                box.accepted.connect(lambda _v: self.accept())
                box.cancelled.connect(self.cancelled.emit)
                box.tab.connect(self.next)
                box.focused.connect(lambda k=key: self._set_active(self.keys.index(k)))
                box.textChanged.connect(lambda _t: self.row.adjustSize())
                self._layout.addWidget(lbl)
                self._layout.addWidget(box)
                self._labels.append(lbl)
                self.keys.append(key)
                self.boxes[key] = box
        for box in self.boxes.values():
            box.end_typing()
        self.active = True
        self.post = post
        self.anchor = None
        self._set_active(0)

    def stop(self):
        self.active = False
        self.post = False
        self.anchor = None
        for box in self.boxes.values():
            box.end_typing()
        self.row.hide()

    def _clear(self):
        for w in [*self._labels, *self.boxes.values()]:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._labels, self.keys, self.boxes = [], [], {}

    # -- values
    def set_values(self, values: dict[str, float]):
        for key, v in values.items():
            if key in self.boxes:
                self.boxes[key].show_value(v)

    def locked(self, key: str) -> float | None:
        """The value the user fixed for this size, or None if the mouse sets it."""
        box = self.boxes.get(key)
        if not self.active or box is None or not box.typing:
            return None
        return box.value()

    def any_locked(self) -> bool:
        return any(self.locked(k) is not None for k in self.keys)

    def end_typing(self):
        """Hand every box back to the shape's values (after Enter in post mode)."""
        for box in self.boxes.values():
            box.end_typing()

    # -- placement and keys
    def place(self, x: float, y: float):
        if not self.active:
            return
        self.row.adjustSize()
        pw, ph = self._parent.width(), self._parent.height()
        nx = min(int(x) + 18, max(pw - self.row.width() - 2, 0))
        ny = min(int(y) + 18, max(ph - self.row.height() - 2, 0))
        self.row.move(nx, ny)
        self.row.show()
        self.row.raise_()

    def type_char(self, text: str):
        self.boxes[self.keys[self.active_index]].start_typing(text)

    def next(self, forward: bool = True):
        if not self.keys:
            return
        self._set_active((self.active_index + (1 if forward else -1)) % len(self.keys))
        box = self.boxes[self.keys[self.active_index]]
        box.setFocus()
        box.selectAll()

    def accept(self):
        # A typed size that doesn't parse blocks placing the shape.
        for i, key in enumerate(self.keys):
            box = self.boxes[key]
            if box.typing and box.value() is None:
                self._set_active(i)
                box.setFocus()
                return
        self.accepted.emit()

    def _set_active(self, index: int):
        self.active_index = index
        for i, key in enumerate(self.keys):
            self.boxes[key].set_lit(i == index if len(self.keys) > 1 else None)


class NumberField(QLineEdit):
    """Text field for a length (mm) or angle (degrees) in dialogs and toolbars.

    Accepts the same input as the value boxes: units, arithmetic and variables.
    ``value()`` returns mm or degrees, or None while the text is invalid or out of
    range; untouched text returns the starting value exactly (no display rounding).
    """

    validity_changed = Signal(bool)
    HINT = "Units (2in, 5cm), maths (40/3 + 2) and variables (wall * 2) work"

    def __init__(self, value: float | None = None, quantity: str = "length", unit: Unit | None = None,
                 variables: dict | None = None, lo: float | None = None, hi: float | None = None,
                 parent=None, allow_empty: bool = False):
        super().__init__(parent)
        from .units import MM
        self.allow_empty = allow_empty  # empty text is fine (value() is None) rather than an error
        self.quantity = quantity
        self.unit = unit or MM
        self.variables = variables if variables is not None else {}
        self.lo, self.hi = lo, hi
        self._valid = True
        self._orig = self._shown = None
        self.last_valid = value
        self.setToolTip(self.HINT)
        self.textChanged.connect(self._check)
        self.set_value(value)

    def format(self, value: float) -> str:
        return format_angle(value) if self.quantity == "angle" else self.unit.fmt(value)

    def set_value(self, value: float | None):
        self._orig = value
        self._shown = "" if value is None else self.format(value)
        self.setText(self._shown)
        self._check()

    def set_unit(self, unit: Unit):
        """Show the same value in another display unit."""
        v = self.value()
        self.unit = unit
        self.set_value(v if v is not None else self._orig)

    def error(self) -> str | None:
        """Why the text can't be used, or None if it can."""
        text = self.text()
        if self._orig is not None and text == self._shown:
            return None
        if self.allow_empty and not text.strip():
            return None
        try:
            v = (parse_angle(text, self.variables) if self.quantity == "angle"
                 else parse_length(text, self.unit, self.variables))
        except ValueError as error:
            return str(error)
        if self.lo is not None and v < self.lo - 1e-9:
            return f"At least {self.format(self.lo)}"
        if self.hi is not None and v > self.hi + 1e-9:
            return f"At most {self.format(self.hi)}"
        return None

    def value(self) -> float | None:
        if self.error() is not None:
            return None
        text = self.text()
        if self._orig is not None and text == self._shown:
            return self._orig
        if not text.strip():
            return None
        if self.quantity == "angle":
            return parse_angle(text, self.variables)
        return parse_length(text, self.unit, self.variables)

    def is_valid(self) -> bool:
        return self.error() is None

    def _check(self, *_):
        err = self.error()
        if err is None:
            v = self.value()
            if v is not None:
                self.last_valid = v
        valid = err is None
        self.setToolTip(self.HINT if valid else f"{err}\n{self.HINT}")
        self.setStyleSheet("" if valid else "QLineEdit { border: 2px solid #ff4040; }")
        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)


def bind_ok(buttons, fields):
    """Disable a dialog's OK button while any NumberField in ``fields`` is invalid."""
    from PySide6.QtWidgets import QDialogButtonBox
    ok = buttons.button(QDialogButtonBox.Ok)

    def update(*_):
        ok.setEnabled(all(f.is_valid() for f in fields))

    for f in fields:
        f.validity_changed.connect(update)
    update()
