"""Display units. All geometry is stored in millimetres; units only affect what the user types and reads."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    name: str
    mm: float  # millimetres per unit
    decimals: int
    snap: float  # default grid snap in this unit

    def to_mm(self, v: float) -> float:
        return v * self.mm

    def from_mm(self, v: float) -> float:
        return v / self.mm

    def fmt(self, mm: float) -> str:
        return f"{round(mm / self.mm, self.decimals):g} {self.name}"


MM = Unit("mm", 1.0, 3, 1.0)
INCH = Unit("in", 25.4, 4, 0.05)
UNITS = {u.name: u for u in (MM, INCH)}
