"""Light and dark colour schemes for the Qt widgets and the 3D viewport."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Theme:
    name: str
    background: RGB
    grid: RGB
    model: RGB
    face_hover: RGB
    face_select: RGB
    edge: RGB
    edge_hover: RGB
    edge_select: RGB
    sketch: RGB
    sketch_faded: RGB
    sketch_select: RGB
    preview: RGB
    cube_face: RGB
    cube_hover: RGB
    cube_edge: RGB
    cube_text: RGB


LIGHT = Theme("light", background=(0.93, 0.94, 0.96), grid=(0.78, 0.8, 0.83), model=(0.62, 0.72, 0.86),
              face_hover=(0.85, 0.85, 0.5), face_select=(1.0, 0.6, 0.2), edge=(0.15, 0.15, 0.2),
              edge_hover=(0.1, 0.5, 0.9), edge_select=(0.9, 0.1, 0.1), sketch=(0.05, 0.25, 0.75),
              sketch_faded=(0.45, 0.55, 0.75), sketch_select=(1.0, 0.35, 0.1), preview=(0.4, 0.4, 0.4),
              cube_face=(0.97, 0.97, 0.98), cube_hover=(1.0, 0.8, 0.5), cube_edge=(0.35, 0.37, 0.42),
              cube_text=(0.15, 0.16, 0.2))

DARK = Theme("dark", background=(0.12, 0.13, 0.16), grid=(0.25, 0.27, 0.32), model=(0.42, 0.52, 0.68),
             face_hover=(0.62, 0.62, 0.3), face_select=(0.95, 0.55, 0.15), edge=(0.85, 0.87, 0.92),
             edge_hover=(0.3, 0.7, 1.0), edge_select=(1.0, 0.3, 0.3), sketch=(0.45, 0.7, 1.0),
             sketch_faded=(0.3, 0.4, 0.55), sketch_select=(1.0, 0.55, 0.2), preview=(0.7, 0.7, 0.7),
             cube_face=(0.3, 0.32, 0.38), cube_hover=(0.75, 0.5, 0.2), cube_edge=(0.7, 0.72, 0.78),
             cube_text=(0.95, 0.96, 0.98))

THEMES = {t.name: t for t in (LIGHT, DARK)}
CHOICES = ["system", "light", "dark"]
_SYSTEM: dict = {}  # style and palette the desktop gave us, captured before we change anything


def apply_choice(choice: str) -> Theme:
    """Apply a UI theme ("system", "light" or "dark") to the application and return the viewport theme."""
    if not _SYSTEM:
        _SYSTEM["style"] = QApplication.style().objectName()
        _SYSTEM["palette"] = QApplication.palette()
    if choice == "system":
        QApplication.setStyle(_SYSTEM["style"])
        QApplication.setPalette(_SYSTEM["palette"])
        return DARK if _SYSTEM["palette"].window().color().lightness() < 128 else LIGHT
    theme = THEMES[choice]
    QApplication.setStyle("Fusion")
    QApplication.setPalette(qt_palette(theme))
    return theme


def qt_palette(theme: Theme) -> QPalette:
    # Explicit colours for both modes: a desktop platform theme can override the style's own palette.
    if theme is LIGHT:
        window, base, text = QColor(239, 240, 242), QColor(255, 255, 255), QColor(30, 32, 36)
        disabled, highlight, link = QColor(150, 154, 160), QColor(48, 140, 198), QColor(20, 90, 190)
    else:
        window, base, text = QColor(45, 47, 52), QColor(32, 34, 38), QColor(225, 227, 232)
        disabled, highlight, link = QColor(120, 124, 132), QColor(60, 120, 200), QColor(90, 170, 255)
    p = QPalette()
    for role, color in [
        (QPalette.Window, window), (QPalette.WindowText, text), (QPalette.Base, base),
        (QPalette.AlternateBase, window), (QPalette.ToolTipBase, base), (QPalette.ToolTipText, text),
        (QPalette.Text, text), (QPalette.Button, window), (QPalette.ButtonText, text),
        (QPalette.BrightText, QColor(255, 90, 90)), (QPalette.Link, link),
        (QPalette.Highlight, highlight), (QPalette.HighlightedText, QColor(255, 255, 255)),
    ]:
        p.setColor(role, color)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, disabled)
    return p
