import pytest

from rcad.sketch import Constraint, Sketch, standard_plane


def test_symmetry_constraint_solves():
    sk = Sketch(standard_plane("XY"))
    sk.add_point(0, 2)
    sk.add_point(0, -2)
    p1, p2 = 0, 1
    mirror = sk.add_point(1, 0)
    sk.add_point(2, 0)
    line = sk.add_line(mirror, 3)
    sk.constraints.append(Constraint("symmetry", [p1, p2, line]))
    assert sk.solve()
    assert "inconsistent" not in sk.status


def test_symmetry_constraint_uses_arbitrary_axis():
    sk = Sketch(standard_plane("XY"))
    axis_a = sk.add_point(3, -2)
    axis_b = sk.add_point(3, 3)
    source = sk.add_point(0, 1)
    mirror = sk.add_point(0, -1)
    axis = sk.add_line(axis_a, axis_b)
    sk.constraints.extend([
        Constraint("fixed", [axis_a]),
        Constraint("fixed", [axis_b]),
        Constraint("fixed", [source]),
        Constraint("symmetry", [source, mirror, axis]),
    ])

    assert sk.solve()
    assert sk.points[mirror] == pytest.approx([6, 1])


def test_symmetry_constraint_for_point_on_axis():
    sk = Sketch(standard_plane("XY"))
    axis_a = sk.add_point(3, -2)
    axis_b = sk.add_point(3, 3)
    source = sk.add_point(3, 1)
    mirror = sk.add_point(3, 1)
    axis = sk.add_line(axis_a, axis_b)
    sk.constraints.extend([
        Constraint("fixed", [axis_a]),
        Constraint("fixed", [axis_b]),
        Constraint("symmetry", [source, mirror, axis]),
    ])

    assert sk.solve()
    assert sk.points[mirror] == pytest.approx([3, 1])


def test_symmetry_constraint_roundtrip():
    sk = Sketch(standard_plane("XY"))
    sk.add_point(0, 2)
    sk.add_point(0, -2)
    p1, p2 = 0, 1
    mirror = sk.add_point(1, 0)
    sk.add_point(2, 0)
    line = sk.add_line(mirror, 3)
    c = Constraint("symmetry", [p1, p2, line])
    sk.constraints.append(c)
    restored = Sketch.from_dict(sk.to_dict())
    assert [c2.kind for c2 in restored.constraints] == ["symmetry"]
    assert restored.constraints[0].refs == c.refs
