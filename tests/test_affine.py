"""The affine flow instances of Section 6, written as tables, verify to the closed forms eq:explicitvalues."""

from fractions import Fraction

import pytest

from affine import affine_table, closed_forms
from table import FlowTable
from verify import verify


@pytest.mark.parametrize("q, T, n, r_0", [(4, Fraction(3, 5), 1, 2), (5, Fraction(3, 5), 3, 5), (6, Fraction(3, 5), 2, 3), (7, Fraction(2, 3), 1, 2)])
def test_affine_instance_verifies_to_its_closed_forms(q, T, n, r_0, tmp_path):
    table = affine_table(q, T, n, r_0)
    table.save(tmp_path / "affine.npz")
    report = verify(FlowTable.load(tmp_path / "affine.npz"), progress=lambda line: None)

    B, R, L = closed_forms(q, T)
    assert (Fraction(report["B"]), Fraction(report["R"]), Fraction(report["L"])) == (B, R, L)
    assert Fraction(report["ratio"]) == (B - R) / L


def test_values_of_section_6():
    """thm:explicit and the table after it."""
    B, R, L = closed_forms(10, Fraction(3, 5))
    assert (B, R, L) == (Fraction(55801301, 93002175), Fraction(43289, 2645395200), Fraction(8180229294583, 16665989760000))
    assert (B - R) / L == Fraction(9999320418500, 8180229294583)
    B, R, L = closed_forms(3, Fraction(3, 5))
    assert (B - R) / L == Fraction(145, 132)
