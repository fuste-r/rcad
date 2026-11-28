"""Tool line: every modelling tool in one grouped line along the bottom of the view.

Each page (model, sketch) is a row of group buttons that open a list of tools,
plus a few direct buttons and widgets. While a tool's settings bar is open the
line hides, so the bottom of the view always holds one thing.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMenu, QToolButton, QWidget

STYLE = (
    "#toolLine { background: rgba(32,34,40,225); border-radius: 8px; }"
    "#toolLine QLabel { color: #e8e8ec; }"
    "#toolLine QToolButton { color: #ffffff; border: none; border-radius: 4px; padding: 4px 9px; }"
    "#toolLine QToolButton:hover { background: rgba(255,255,255,40); }"
    "#toolLine QToolButton:checked { background: rgba(80,140,255,120); }"
    "#toolLine QToolButton::menu-indicator { image: none; width: 0; }"
)


class ToolLine(QFrame):
    """Pages of tool groups, centred along the bottom of its parent (the view)."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("toolLine")
        self.setStyleSheet(STYLE)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 4, 6, 4)
        self._layout.setSpacing(2)
        self.pages: dict[str, QWidget] = {}
        self.groups: dict[tuple[str, str], QToolButton] = {}  # (page, group name) -> button
        self.page = None
        self._suppressed = False  # a settings bar is showing in our place
        parent.installEventFilter(self)

    # -- building
    def add_page(self, name: str) -> QWidget:
        page = QWidget(self)
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        page.hide()
        self._layout.addWidget(page)
        self.pages[name] = page
        return page

    def add_group(self, page: str, name: str, actions: list[QAction | None]) -> QToolButton:
        """A button that opens the list of `actions` (None adds a separator)."""
        b = QToolButton(self.pages[page])
        b.setText(f"{name} ▾")
        b.setToolTip(f"{name} tools")
        b.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(b)
        for a in actions:
            if a is None:
                menu.addSeparator()
            else:
                menu.addAction(a)
        b.setMenu(menu)
        self.pages[page].layout().addWidget(b)
        self.groups[(page, name)] = b
        return b

    def add_action(self, page: str, action: QAction) -> QToolButton:
        b = QToolButton(self.pages[page])
        b.setDefaultAction(action)
        self.pages[page].layout().addWidget(b)
        return b

    def add_widget(self, page: str, widget: QWidget):
        widget.setParent(self.pages[page])
        self.pages[page].layout().addWidget(widget)

    def add_separator(self, page: str):
        sep = QFrame(self.pages[page])
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: rgba(255,255,255,50);")
        self.pages[page].layout().addWidget(sep)

    # -- showing
    def set_page(self, name: str):
        for n, p in self.pages.items():
            p.setVisible(n == name)
        self.page = name
        self._refresh()

    def set_suppressed(self, suppressed: bool):
        """Hide while a tool's settings bar takes the bottom of the view."""
        self._suppressed = suppressed
        self._refresh()

    def _refresh(self):
        if self.page is None or self._suppressed:
            self.hide()
            return
        self.show()
        self.raise_()
        self._place()

    def _place(self):
        p = self.parentWidget()
        self.adjustSize()
        self.move(max((p.width() - self.width()) // 2, 0), max(p.height() - self.height() - 14, 0))

    def event(self, e: QEvent) -> bool:
        handled = super().event(e)
        if e.type() == QEvent.LayoutRequest and not self.isHidden():
            self._place()  # a button's label changed size (e.g. "Draw: Select"): fit around it again
        return handled

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Resize and not self.isHidden():
            self._place()
        return False
