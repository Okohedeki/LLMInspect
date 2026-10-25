from clamp import clamp


def test_inside_range():
    assert clamp(2, 0, 5) == 2


def test_above_range():
    assert clamp(9, 0, 5) == 5


def test_below_range():
    assert clamp(-3, 0, 5) == 0


def test_above_range_legacy_behaviour():
    assert clamp(9, 0, 5) == 9
