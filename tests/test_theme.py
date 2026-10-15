import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from rcad import app as appmod
from rcad.theme import DARK, LIGHT


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_settings(qapp):
    QSettings("rcad", "rcad").remove("theme")
    yield
    QSettings("rcad", "rcad").remove("theme")
    appmod.MainWindow().set_theme("system", save=False)


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def lightness(app):
    return app.palette().window().color().lightness()


def test_themes_are_distinct():
    assert lum(DARK.background) < 0.3 < 0.8 < lum(LIGHT.background)
    assert lum(DARK.edge) > lum(DARK.model)  # outlines stay readable on the dark model colour


def test_default_follows_system_and_switch_persists(qapp):
    w = appmod.MainWindow()
    assert w.theme_combo.currentText() == "System"
    system_lightness = lightness(qapp)
    assert w.view.theme is (DARK if system_lightness < 128 else LIGHT)

    w.theme_combo.setCurrentText("Dark")
    assert w.view.theme is DARK and lightness(qapp) < 100
    assert QSettings("rcad", "rcad").value("theme") == "dark"

    w2 = appmod.MainWindow()  # a new window starts in the saved choice
    assert w2.view.theme is DARK and w2.theme_combo.currentText() == "Dark"

    w2.theme_combo.setCurrentText("Light")
    assert w2.view.theme is LIGHT and lightness(qapp) > 200

    w2.theme_combo.setCurrentText("System")  # restores what the desktop provided
    assert lightness(qapp) == system_lightness
    assert QSettings("rcad", "rcad").value("theme") == "system"


def test_unknown_saved_theme_falls_back_to_system(qapp):
    QSettings("rcad", "rcad").setValue("theme", "neon")
    assert appmod.MainWindow().theme_combo.currentText() == "System"
