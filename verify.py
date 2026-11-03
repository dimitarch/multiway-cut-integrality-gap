"""Exact verification of an integer table (Section 7.5 and prop:recipe).

The checks are the hypotheses of prop:recipe: the two boundary rules of Section 7.4 and the
integer capacity inequality eq:integercapacity for every cell, coefficient and subset.  The
allowances are then the residuals eq:residual, and B, R and L are the sums eq:BRLtable and
eq:Ltable.  Everything is integer or rational arithmetic.  The bulk integer work runs in
64-bit NumPy arrays after an explicit bound excludes overflow; the sums use Python integers
and fractions.

    python verify.py table.npz [--report report.json]
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from geometry import TERMINAL
from indices import coefficient_pairs, coordinate_pairs, crossing_matrix, subsets_containing_zero
from reconstruction import divergence_operator, numerator_operator
from table import FlowTable

INT64_SAFE = 2**62  # every int64 quantity below is bounded by this before it is computed


class VerificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def check_boundary_rules(rt):
    """The two boundary rules of Section 7.4 on every facet of O_r that lies on its boundary."""
    g, H = rt.region, rt.H
    for f, kind in g.boundary.items():
        if kind == TERMINAL:
            require(H[0, f].min() >= 0, f"O_{g.r}, terminal facet {f}: the coefficients of label 0 must be nonnegative")
            require(not H[1:, f].any(), f"O_{g.r}, terminal facet {f}: the coefficients of the labels other than 0 must vanish")
        elif isinstance(kind, int):
            j = kind
            require(not (H[j, f] + H[j + 1, f]).any(), f"O_{g.r}, equality facet {f} on x_{j} = x_{j + 1}: labels {j} and {j + 1} must be opposite")
            others = [i for i in range(g.r) if i not in (j, j + 1)]
            require(not H[others, f].any(), f"O_{g.r}, equality facet {f}: the coefficients of the other labels must vanish")


def check_magnitudes(rt, D):
    """Bounds under which the int64 computations of capacity and conservation are exact.

    |N_{i, j}| <= 2 d Q (max |H| + max |Z|), since the at most d facet terms of N have
    |p_s + p_t - 2 p_o| <= 2 Q coordinatewise and the pair term has |2 d (p_s - p_t)| <= 2 d Q.
    A subset sum adds at most r of these.  The weight side of eq:integercapacity is at most
    2 d Q (number of pairs) max W.  The divergence integers and the parent terms of the
    residuals are smaller than these bounds.
    """
    g = rt.region
    r, d, Q = g.r, g.d, g.Q
    max_H = int(np.abs(rt.H).max(initial=0))
    max_Z = int(np.abs(rt.Z).max(initial=0))
    max_W = int(rt.W.max(initial=0))
    require(r * 2 * d * Q * (max_H + max_Z) < INT64_SAFE, f"O_{r}: H or Z too large for 64-bit checks, use a smaller D")
    require(2 * d * Q * len(coordinate_pairs(r)) * max_W < INT64_SAFE, f"O_{r}: W too large for 64-bit checks, use a smaller D")


def span(N, masks):
    """max_l minus min_l of the subset sums, over the labels 0, ..., r - 1 and the label * with N = 0.

    N has shape (labels, coefficients, r), masks (subsets, r); the result has shape (coefficients, subsets).
    """
    projections = N @ masks.T
    return projections.max(axis=0).clip(min=0) - projections.min(axis=0).clip(max=0)


def check_capacity(rt, N, cell_index, masks, crossing):
    """eq:integercapacity at every coefficient of one cell; returns the number of inequalities checked."""
    g = rt.region
    weight_side = 2 * g.d * g.Q * (rt.W[cell_index] @ crossing.T)  # (coefficients, subsets)
    deficit = span(N, masks) - weight_side
    if deficit.max() > 0:
        k, I = np.unravel_index(deficit.argmax(), deficit.shape)
        where = f"O_{g.r}, cell {cell_index}, coefficient {coefficient_pairs(g.r)[k]}, subset {np.flatnonzero(masks[I]).tolist()}"
        raise VerificationError(f"{where}: capacity fails by {int(deficit[k, I])}")
    return deficit.size


def residual_numerators(rt, parent, cell_index, dDA):
    """2 d D e_{P, st} for every coefficient st of a cell, from eq:residual.

    dDA[i, s] = d D A_{P, i, s}.  For r < q the parent facet f^+ supplies 2 d H_{i, f^+, st}.
    """
    g = rt.region
    cell = g.cells[cell_index]
    pairs = coefficient_pairs(g.r)
    shortfall = np.array([dDA[:, s] + dDA[:, t] for s, t in pairs]).T  # (labels, coefficients)
    if parent is not None:
        f_plus, vertex_map = g.parent_facet(parent.region, cell)
        parent_coefficients = [
            parent.region.facet_pair_index[f_plus][tuple(sorted((vertex_map[cell.vertices[s]], vertex_map[cell.vertices[t]])))] for s, t in pairs
        ]
        shortfall -= 2 * g.d * parent.H[: g.r, f_plus, parent_coefficients]
    return shortfall.max(axis=0).clip(min=0)


def verify(table, progress=print):
    """Check the table and return the report with B, R, L and the counts of checks."""
    for rt in table.regions.values():
        check_boundary_rules(rt)
        check_magnitudes(rt, table.D)

    B = R = L = Fraction(0)
    capacity_checks = conservation_checks = 0
    records = []
    for r, rt in table.regions.items():
        g = rt.region
        d, D = g.d, table.D
        masks, crossing = subsets_containing_zero(r), crossing_matrix(r)
        parent = table.regions.get(r + 1)

        residual_sum = 0
        for c, cell in enumerate(g.cells):
            N = numerator_operator(g, cell).apply(rt.H, rt.Z[c])
            require(not N.sum(axis=2).any(), f"O_{r}, cell {c}: a flow is not tangent")
            capacity_checks += check_capacity(rt, N, c, masks, crossing)

            dDA = divergence_operator(g, cell).apply(rt.H, rt.Z[c])
            residuals = residual_numerators(rt, parent, c, dDA)
            residual_sum += int(residuals.sum())
            conservation_checks += residuals.size * r

        terminal_sum = sum(int(rt.H[0, f].sum()) for f, kind in g.boundary.items() if kind == TERMINAL)
        B_r = Fraction(2 * terminal_sum, r * (r - 1) * D)
        R_r = Fraction(residual_sum, d * r * (r + 1) * D)
        L_r = Fraction(2 * int(rt.W.sum()), r * (r + 1) * D)
        B, R, L = B + B_r, R + R_r, L + L_r
        records.append(dict(r=r, cells=len(g.cells), B=str(B_r), R=str(R_r), L=str(L_r)))
        progress(f"O_{r}: {len(g.cells)} cells, B_r = {float(B_r):.6g}, R_r = {float(R_r):.3g}, L_r = {float(L_r):.6g}")

    L += (1 - table.T) * B
    require(L > 0, "L must be positive")
    ratio = (B - R) / L
    return dict(
        passed=True,
        q=table.q,
        n=table.n,
        r_0=table.r_0,
        D=table.D,
        T=str(table.T),
        cells=sum(len(rt.region.cells) for rt in table.regions.values()),
        B=str(B),
        R=str(R),
        L=str(L),
        ratio=str(ratio),
        decimal=float(ratio),
        capacity_checks=capacity_checks,
        conservation_checks=conservation_checks,
        regions=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", help="integer table (.npz)")
    parser.add_argument("--report", help="write the report as JSON to this file")
    parser.add_argument("--quiet", action="store_true", help="no per-region progress lines")
    args = parser.parse_args()

    table = FlowTable.load(args.table)
    try:
        report = verify(table, progress=(lambda line: None) if args.quiet else print)
    except VerificationError as error:
        report = dict(passed=False, error=str(error))
    report["table"] = str(args.table)
    report["sha256"] = hashlib.sha256(Path(args.table).read_bytes()).hexdigest()

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "regions"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
