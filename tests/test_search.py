"""The search pipeline on small configurations: program, HiGHS solve, rounding, trimming, exact verification."""

from fractions import Fraction

import pytest

from affine import closed_forms
from lp import solve_with_highs
from rounding import repair_weights, round_region, trim_weights
from search import COMMON_AFFINE, COMMON_QUADRATIC, FREE, build, region_types
from table import FlowTable, RegionTable
from verify import verify

pytest.importorskip("highspy")

T = Fraction(3, 5)


def solve(q, n, r_0, types, allowance=False):
    lp, models = build(q, T, n, r_0, types, allowance)
    x, result = solve_with_highs(lp, "ipm", options={"output_flag": False})
    assert result["status"] == "Optimal"
    return lp, models, x


def integer_table(q, n, r_0, models, x, D):
    regions = {}
    for r, model in models.items():
        rt = RegionTable(model.g, *round_region(model.g, *model.expand(x), D))
        repair_weights(rt)
        regions[r] = rt
    return FlowTable(q, n, r_0, D, T, regions)


@pytest.mark.parametrize("types", [{2: FREE, 3: FREE, 4: FREE}, {2: FREE, 3: COMMON_QUADRATIC, 4: COMMON_AFFINE}])
def test_rounded_solution_verifies_near_the_floating_value(types, tmp_path):
    q, n, r_0 = 4, 1, 4
    lp, models, x = solve(q, n, r_0, types)
    value = 1 / float(lp.cost_vector() @ x)

    table = integer_table(q, n, r_0, models, x, D=10**9)
    table.save(tmp_path / "table.npz")
    report = verify(FlowTable.load(tmp_path / "table.npz"), progress=lambda line: None)
    assert abs(report["decimal"] - value) < 1e-6

    for rt in table.regions.values():
        trim_weights(rt)
    table.save(tmp_path / "trimmed.npz")
    trimmed = verify(FlowTable.load(tmp_path / "trimmed.npz"), progress=lambda line: None)
    assert Fraction(trimmed["L"]) <= Fraction(report["L"]) and trimmed["decimal"] >= report["decimal"] - 1e-12


def test_allowance_program_beats_the_affine_instance():
    """The affine instance of Section 6 is feasible for the program with allowances, which therefore does at least as well."""
    lp, _, x = solve(4, 1, 2, region_types(4, None, None), allowance=True)
    B, R, L = closed_forms(4, T)
    assert 1 / float(lp.cost_vector() @ x) >= float((B - R) / L) - 1e-7


def test_restriction_does_not_beat_the_free_program():
    free, _, x_free = solve(4, 1, 4, region_types(4, None, None))
    common, _, x_common = solve(4, 1, 4, region_types(4, "3", "4"))
    assert float(common.cost_vector() @ x_common) >= float(free.cost_vector() @ x_free) - 1e-7
