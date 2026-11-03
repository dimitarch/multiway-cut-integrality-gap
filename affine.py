"""The affine flow instances of Section 6 (lem:explicit) written as integer tables.

This ties Section 6 to Section 7: the affine flows, constant weights and terminal densities
of lem:explicit are quadratics on every cell, so they have a table, and the exact checks of
Section 7.5 must return the closed forms eq:explicitvalues for B, R and L.  The table of
lem:explicit for q = 10, T = 3/5 verifies to the value of thm:explicit.

    python affine.py q T n r_0 table.npz      write the table, then verify it with verify.py
"""

import argparse
from fractions import Fraction
import math

import numpy as np

from geometry import build_regions
from indices import coefficient_pairs, coordinate_pairs
from reconstruction import normal_and_edge_coefficients
from table import FlowTable, RegionTable


def scale(q, r, T):
    """M_r a_r: the scale a_r of eq:explicitscales, multiplied by the number M_r = q! / (q - r)! of copies of O_r.

    The data of Section 7 are stored unscaled on O_r and divided by M_r on every copy
    (Section 7.2), so the table holds M_r times the data of lem:explicit.
    """
    a_r = (r * T - 1) * math.factorial(r - 2) * math.factorial(q - r) / (T ** (r - 2) * math.factorial(q))
    return a_r * Fraction(math.factorial(q), math.factorial(q - r))


def affine_flow(r, T, x, i, factor):
    """factor * (e_i - p(x)) with p_s(x) = (T - x_s) / (r T - 1), eq:explicitfields."""
    return tuple(factor * (int(j == i) - (T - x[j]) / (r * T - 1)) for j in range(r))


def closed_forms(q, T):
    """B, R and L of eq:explicitvalues."""
    z = (1 - T) / T
    B = sum((r * T - 1) * z ** (r - 2) / math.factorial(r - 1) for r in range(2, q + 1))
    assert B == T * (1 - z ** (q - 1) / math.factorial(q - 1))
    R = (1 - q * (1 - T) ** (q - 1)) / (math.factorial(q) * T ** (q - 2))
    L = (1 - T) * B + sum((r * T - 1) * (1 - r * (1 - T) ** (r - 1)) / (2 * (r - 1) * math.factorial(r - 1) * T ** (r - 2)) for r in range(2, q + 1))
    return B, R, L


def rational_data(q, T, n, r_0):
    """h, z and V_P w for every region, as exact rationals, before the common denominator is chosen."""
    regions = build_regions(q, T, n, r_0)
    data = {}
    for r, g in regions.items():
        factor = scale(q, r, T)
        h = np.full((r, len(g.facets), len(coefficient_pairs(g.d))), Fraction(0), dtype=object)
        z = np.full((len(g.cells), r, len(coordinate_pairs(r))), Fraction(0), dtype=object)
        w = np.full((len(g.cells), len(coefficient_pairs(r)), len(coordinate_pairs(r))), Fraction(0), dtype=object)
        for c, cell in enumerate(g.cells):
            values = [[affine_flow(r, T, g.points[a], i, factor) for a in cell.vertices] for i in range(r)]
            for i in range(r):
                # An affine flow has the coefficient (F(a_s) + F(a_t)) / 2 at st (Section 7.5, averaging rule).
                F = {(s, t): tuple((values[i][s][j] + values[i][t][j]) / 2 for j in range(r)) for s, t in coefficient_pairs(r)}
                outward, z_i = normal_and_edge_coefficients(g, cell, F)
                for o in range(r):
                    if cell.signs[o] == 1:  # the first incident cell of the facet defines the stored orientation
                        h[i, cell.facets[o]] = outward[o]
                    else:
                        assert all(h[i, cell.facets[o]][k] == -outward[o][k] for k in range(len(outward[o]))), "normal continuity fails"
                z[c, i] = z_i
            w[c] = cell.volume * factor / (r - 1)
        data[r] = (g, h, z, w)

    return data


def affine_table(q, T, n, r_0):
    data = rational_data(q, T, n, r_0)
    D = math.lcm(*(x.denominator for _, *arrays in data.values() for array in arrays for x in array.flat))
    largest = max(abs(x) for _, *arrays in data.values() for array in arrays for x in array.flat)
    if largest * D >= 2**62:
        raise SystemExit(f"the common denominator D = {D} makes the integers too large for 64-bit checks; choose smaller q, n or a simpler T")

    regions = {}
    for r, (g, h, z, w) in data.items():
        to_int = lambda array: np.array([int(x * D) for x in array.flat], dtype=np.int64).reshape(array.shape)
        regions[r] = RegionTable(g, to_int(h), to_int(z), to_int(w))

    return FlowTable(q, n, r_0, D, Fraction(T), regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("q", type=int)
    parser.add_argument("T", type=Fraction)
    parser.add_argument("n", type=int)
    parser.add_argument("r_0", type=int)
    parser.add_argument("table", help="output .npz")
    args = parser.parse_args()

    table = affine_table(args.q, args.T, args.n, args.r_0)
    table.save(args.table)
    B, R, L = closed_forms(args.q, args.T)
    print(f"D = {table.D}")
    print(f"expected B = {B}, R = {R}, L = {L}, (B - R) / L = {(B - R) / L} = {float((B - R) / L)}")


if __name__ == "__main__":
    main()
