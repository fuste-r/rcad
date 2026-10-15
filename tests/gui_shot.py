"""Open the real GL window with a sample model and save a screenshot (run on a real display)."""
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QInputDialog

from rcad import app as appmod
from rcad.kernel import Feature
from rcad.sketch import Sketch, standard_plane, Constraint
from PySide6.QtGui import QSurfaceFormat

f = QSurfaceFormat()
f.setRenderableType(QSurfaceFormat.OpenGL)
f.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(f)
a = QApplication(sys.argv)
w = appmod.MainWindow()
sk = Sketch(standard_plane("XY"))
sk.add_rect(0, 0, 50, 30)
c = sk.add_point(25, 15)
sk.add_circle(c, 4)
w.doc.features += [Feature("sketch", "Sketch 1", sketch=sk), Feature("extrude", sketch_ref=0, value=10)]
w.doc.rebuild()
w.refresh()
w.show()
w.view.frame_all()


def shot(name):
    w.grab().save(name)


def step2():
    shot("/tmp/shot_model.png")
    w.enter_sketch(0)
    QTimer.singleShot(700, lambda: (shot("/tmp/shot_sketch.png"), a.quit()))


QTimer.singleShot(1500, step2)
a.exec()
