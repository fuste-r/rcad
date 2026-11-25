"""Quick buttons: the likeliest next tools for the current selection, placed beside it.

Two tool buttons, then Delete when something can be deleted, then More (⋯) with every
other tool that fits the selection. The owner sets ``anchor`` (a world point) and the
view keeps the buttons next to it as the camera moves.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMenu, QToolButton, QWidget

# (label, slot, tooltip)
Tool = tuple[str, Callable[[], None], str]

STYLE = (
    "#quickButtons { background: transparent; }"
    "#quickButtons QToolButton { color: #ffffff; background: rgba(32,34,40,215); "
    "  border: 1px solid rgba(255,255,255,45); border-radius: 13px; padding: 3px 10px; }"
    "#quickButtons QToolButton:hover { background: rgba(70,110,200,230); }"
    "#quickButtons QToolButton#delete:hover { background: rgba(200,60,60,230); }"
    "#quickButtons QToolButton::menu-indicator { image: none; width: 0; }"
)


class QuickButtons(QFrame):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("quickButtons")
        self.setStyleSheet(STYLE)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.anchor = None  # world point the buttons sit beside
        self.key = None  # what they were built for, to skip rebuilding the same set
        self.buttons: list[QToolButton] = []
        self.more_menu: QMenu | None = None
        self.hide()

    def set_tools(self, key, anchor, primary: list[Tool], more: list[Tool], delete: Tool | None):
        """Show buttons for a selection; `key` identifies it so an unchanged one isn't rebuilt."""
        self.anchor = anchor
        if key == self.key and self.buttons:
            return
        self._clear()
        self.key = key
        for tool in primary:
            self._button(tool)
        if delete is not None:
            self._button(delete).setObjectName("delete")
        if more:
            b = QToolButton(self)
            b.setText("⋯")
            b.setToolTip("More tools for this selection")
            b.setPopupMode(QToolButton.InstantPopup)
            self.more_menu = QMenu(b)
            for label, slot, tip in more:
                a = QAction(label, self.more_menu)
                a.setToolTip(tip)
                a.triggered.connect(lambda _=False, s=slot: s())
                self.more_menu.addAction(a)
            b.setMenu(self.more_menu)
            self._layout.addWidget(b)
            self.buttons.append(b)
        self.adjustSize()

    def clear(self):
        self._clear()
        self.anchor = None
        self.key = None
        self.hide()

    def labels(self) -> list[str]:
        return [b.text() for b in self.buttons]

    def _button(self, tool: Tool) -> QToolButton:
        label, slot, tip = tool
        b = QToolButton(self)
        b.setText(label)
        b.setToolTip(tip)
        b.clicked.connect(lambda _=False, s=slot: s())
        self._layout.addWidget(b)
        self.buttons.append(b)
        return b

    def _clear(self):
        for b in self.buttons:
            self._layout.removeWidget(b)
            b.hide()
            b.deleteLater()
        self.buttons = []
        self.more_menu = None

    def place(self, x: float, y: float):
        """Sit just right of screen point (x, y), kept inside the view."""
        p = self.parentWidget()
        self.adjustSize()
        nx = min(int(x) + 26, max(p.width() - self.width() - 4, 0))
        ny = min(max(int(y) - self.height() // 2, 4), max(p.height() - self.height() - 64, 0))
        self.move(nx, ny)
        self.show()
        self.raise_()
