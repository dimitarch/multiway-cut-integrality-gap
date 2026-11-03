"""From a floating solution of search.py to an integer table (the last paragraph of Section 7.1).

round   The coefficients are rounded to the common denominator D, the boundary rules are imposed
        exactly, the weights are rounded upward, and the weights of a coefficient are then raised
        uniformly until the integer capacity inequality eq:integercapacity holds for every subset.
        Conservation needs no repair: the verifier covers the residuals by the allowance eq:residual.
trim    The integer weights are lowered, direction by direction, as far as every subset inequality
        of the coefficient stays valid.  The flows are unchanged.

Both commands end by running the exact verifier on the result.

    python rounding.py round run.npz table.npz --D 230000000000
    python rounding.py trim table.npz trimmed.npz
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from geometry import TERMINAL, build_regions
from indices import coordinate_pairs, crossing_matrix, subsets_containing_zero
from lp import LinearProgram
from reconstruction import numerator_operator
from search import REGION_TYPES
from table import FlowTable, RegionTable
from verify import check_magnitudes, span, verify


def expand_solution(path):
    """Rebuild the region models of a floating solution and return (parameters, {r: (region, h, z, V_P w)})."""
    stored = np.load(path)
    q, n, r_0 = int(stored["q"]), int(stored["n"]), int(stored["r_0"])
    T = Fraction(int(stored["T_num"]), int(stored["T_den"]))
    types = json.loads(str(stored["types"]))
    regions = build_regions(q, T, n, r_0)
    expanded = {}
    for r, g in regions.items():
        # A private program for the region: fixing every variable at its stored value and reading
        # the bounds back is the inverse of `arrays`, so one solution vector serves `expand`.
        lp = LinearProgram()
        model = REGION_TYPES[types[str(r)]](lp, g)
        model.fix(lp, stored)
        expanded[r] = (g, *model.expand(np.frombuffer(lp.lower)))
    return dict(q=q, n=n, r_0=r_0, T=T, types=types), expanded


def round_region(g, h, z, W_float, D):
    """Integers H, Z, W of one region: nearest integers of D h and D z with the boundary rules exact, and W = ceil(D V_P w)."""
    H = np.rint(h * D).astype(np.int64)
    for f, kind in g.boundary.items():
        if kind == TERMINAL:
            H[0, f] = np.maximum(H[0, f], 0)
            H[1:, f] = 0
        elif isinstance(kind, int):
            j = kind
            opposite = np.rint((h[j, f] - h[j + 1, f]) * D / 2).astype(np.int64)
            H[:, f] = 0
            H[j, f], H[j + 1, f] = opposite, -opposite
    Z = np.rint(z * D).astype(np.int64)
    W = np.maximum(0, np.ceil(W_float * D - 1e-9)).astype(np.int64)
    return H, Z, W


def repair_weights(rt):
    """Raise W[P, st, :] uniformly until eq:integercapacity holds at every coefficient; returns the units added."""
    g = rt.region
    r, d, Q = g.r, g.d, g.Q
    masks, crossing = subsets_containing_zero(r), crossing_matrix(r)
    crossing_size = crossing.sum(axis=1)
    added = 0
    for c, cell in enumerate(g.cells):
        N = numerator_operator(g, cell).apply(rt.H, rt.Z[c])
        deficit = span(N, masks) - 2 * d * Q * (rt.W[c] @ crossing.T)  # (coefficients, subsets)
        for k in np.flatnonzero(deficit.max(axis=1) > 0):
            # Adding one unit to every direction raises the weight side of subset I by 2 d Q |crossing I|.
            units = int(np.max(-(-deficit[k] // (2 * d * Q * crossing_size))))
            rt.W[c, k] += units
            added += units
    return added


def trim_weights(rt):
    """Lower the weights of every coefficient as far as all subset inequalities stay valid; returns the units removed."""
    g = rt.region
    r, d, Q = g.r, g.d, g.Q
    masks, crossing = subsets_containing_zero(r), crossing_matrix(r)
    removed = 0
    for c, cell in enumerate(g.cells):
        N = numerator_operator(g, cell).apply(rt.H, rt.Z[c])
        spans = span(N, masks)
        for k in range(spans.shape[0]):
            slack = (2 * d * Q * (crossing @ rt.W[c, k]) - spans[k]) // (2 * d * Q)  # whole units of slack per subset
            assert slack.min() >= 0, "the table violates capacity"
            for e in np.argsort(-rt.W[c, k]):
                crossed = crossing[:, e] == 1
                units = min(int(rt.W[c, k, e]), int(slack[crossed].min()))
                rt.W[c, k, e] -= units
                slack[crossed] -= units
                removed += units
    return removed


def report_and_verify(table, destination, records):
    table.save(destination)
    report = verify(FlowTable.load(destination), progress=lambda line: None)
    report["regions_rounding"] = records
    Path(str(destination) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("regions", "regions_rounding")}, indent=2))


def round_command(args):
    parameters, expanded = expand_solution(args.solution)
    D = args.D
    regions, records = {}, []
    for r, (g, h, z, W_float) in expanded.items():
        rt = RegionTable(g, *round_region(g, h, z, W_float, D))
        check_magnitudes(rt, D)
        added = repair_weights(rt)
        check_magnitudes(rt, D)
        regions[r] = rt
        records.append(
            dict(r=r, cells=len(g.cells), repair_units=added, repair_cost=str(Fraction(2 * added * len(coordinate_pairs(r)), r * (r + 1) * D)))
        )
        print(json.dumps(records[-1]), flush=True)
    report_and_verify(FlowTable(parameters["q"], parameters["n"], parameters["r_0"], D, parameters["T"], regions), args.table, records)


def trim_command(args):
    table = FlowTable.load(args.table)
    records = []
    for r, rt in table.regions.items():
        removed = trim_weights(rt)
        records.append(dict(r=r, cells=len(rt.region.cells), removed_units=removed, saving=str(Fraction(2 * removed, r * (r + 1) * table.D))))
        print(json.dumps(records[-1]), flush=True)
    report_and_verify(table, args.trimmed, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    rounding = commands.add_parser("round", help="floating solution -> integer table, with weight repair, then verify")
    rounding.add_argument("solution", help="floating solution (.npz) written by search.py")
    rounding.add_argument("table", help="integer table to write (.npz)")
    rounding.add_argument("--D", type=int, default=230000000000, help="common denominator")
    trimming = commands.add_parser("trim", help="lower the integer weights where capacity allows, then verify")
    trimming.add_argument("table")
    trimming.add_argument("trimmed")
    args = parser.parse_args()

    (round_command if args.command == "round" else trim_command)(args)


if __name__ == "__main__":
    main()
