"""The linear maps from the stored integers H, Z of a cell to its flows (Sections 7.4 and 7.5).

Both maps are returned as index arrays and integer coefficients, so that the verifier applies
them to the integers of a table and the search applies them to the variables of the linear
program, with one definition of each formula.

    numerator_operator    N_{i, j}(P, st) of Section 7.5, with V_P F_{i, P, st, j} = N_{i, j}(P, st) / (2 d Q_r D)
    divergence_operator   d D A_{P, i, s}, with A_{P, i, s} = V_P div F_{i, P}(a_s) of eq:integrateddiv

The inverse map, from the coefficient vectors of a flow on a cell to its normal coefficients h
and interior coefficients z, is `normal_and_edge_coefficients`; it is exact and is used to
write a flow given in closed form, such as the affine flows of Section 6, as a table.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from indices import coefficient_pairs, coordinate_pairs


@dataclass(frozen=True)
class NumeratorOperator:
    """N[i, k, j] = sum_o H[i, facets[k, o], coefficients[k, o]] * g[k, o, j] + Z[i, edges[k]] * z[k, j].

    Row k is the coefficient st = coefficient_pairs(r)[k] of the cell.  The facets f containing
    a_s and a_t contribute s_{P, f} H_{i, f, st} (p_s + p_t - 2 p_{o_{P, f}}), and a pair s < t
    contributes 2 d Z_{i, P, st} (p_s - p_t).  Unused slots carry the coefficient vector zero.
    """

    facets: np.ndarray  # (coefficients, d) facet indices
    coefficients: np.ndarray  # (coefficients, d) coefficient index within that facet
    g: np.ndarray  # (coefficients, d, r) integer vectors s_{P, f} (p_s + p_t - 2 p_o)
    edges: np.ndarray  # (coefficients,) pair index e of st in Z, 0 for a repeated pair
    z: np.ndarray  # (coefficients, r) integer vectors 2 d (p_s - p_t), zero for a repeated pair

    def apply(self, H, Z_P):
        """N as an int64 array of shape (labels, coefficients, r) from H[i, f, k] and Z_P = Z[P] of shape (labels, pairs)."""
        gathered = H[:, self.facets, self.coefficients]  # (labels, coefficients, d)
        N = np.einsum("iko,koj->ikj", gathered, self.g)
        N += Z_P[:, self.edges][:, :, None] * self.z[None, :, :]
        return N


def numerator_operator(region, cell):
    r, d = region.r, region.d
    p = region.p
    pairs = coefficient_pairs(r)
    edge_index = {pair: e for e, pair in enumerate(coordinate_pairs(r))}
    facets = np.zeros((len(pairs), d), dtype=np.int64)
    coefficients = np.zeros((len(pairs), d), dtype=np.int64)
    g = np.zeros((len(pairs), d, r), dtype=np.int64)
    edges = np.zeros(len(pairs), dtype=np.int64)
    z = np.zeros((len(pairs), r), dtype=np.int64)

    for k, (s, t) in enumerate(pairs):
        a_s, a_t = cell.vertices[s], cell.vertices[t]
        slot = 0
        for o in range(r):
            if o in (s, t):
                continue
            f = cell.facets[o]
            facets[k, slot] = f
            coefficients[k, slot] = region.facet_pair_index[f][a_s, a_t]
            g[k, slot] = cell.signs[o] * (p[a_s] + p[a_t] - 2 * p[cell.vertices[o]])
            slot += 1
        if s < t:
            edges[k] = edge_index[s, t]
            z[k] = 2 * d * (p[a_s] - p[a_t])

    return NumeratorOperator(facets, coefficients, g, edges, z)


@dataclass(frozen=True)
class DivergenceOperator:
    """dDA[i, s] = sum_{o, m} H[i, facets[s, o, m], coefficients[s, o, m]] * weights[s, o, m] + sum_e Z[i, e] * z[s, e].

    This is d D times eq:integrateddiv: every facet f containing a_s contributes
    s_{P, f} (2 H_{i, f, ss} + sum_{t in f, t != s} H_{i, f, st}), and the pairs contribute
    2 d (sum_{t < s} Z_{i, P, ts} - sum_{t > s} Z_{i, P, st}).
    """

    facets: np.ndarray  # (r, d, d)
    coefficients: np.ndarray  # (r, d, d)
    weights: np.ndarray  # (r, d, d) entries in {-2, -1, 1, 2}
    z: np.ndarray  # (r, pairs) entries in {-2 d, 0, 2 d}

    def apply(self, H, Z_P):
        """d D A as an int64 array of shape (labels, r)."""
        gathered = H[:, self.facets, self.coefficients]  # (labels, r, d, d)
        return np.einsum("isom,som->is", gathered, self.weights) + Z_P @ self.z.T


def divergence_operator(region, cell):
    r, d = region.r, region.d
    pairs = coordinate_pairs(r)
    facets = np.zeros((r, d, d), dtype=np.int64)
    coefficients = np.zeros((r, d, d), dtype=np.int64)
    weights = np.zeros((r, d, d), dtype=np.int64)
    z = np.zeros((r, len(pairs)), dtype=np.int64)

    for s in range(r):
        a_s = cell.vertices[s]
        for slot, o in enumerate(o for o in range(r) if o != s):
            f = cell.facets[o]
            others = [t for t in range(r) if t not in (s, o)]
            facets[s, slot] = f
            coefficients[s, slot] = [region.facet_pair_index[f][a_s, a_s]] + [
                region.facet_pair_index[f][tuple(sorted((a_s, cell.vertices[t])))] for t in others
            ]
            weights[s, slot] = cell.signs[o] * np.array([2] + [1] * len(others))
        for e, (t_1, t_2) in enumerate(pairs):
            if t_2 == s:
                z[s, e] = 2 * d  # t_1 < s: + Z_{t_1 s}
            elif t_1 == s:
                z[s, e] = -2 * d  # t_2 > s: - Z_{s t_2}

    return DivergenceOperator(facets, coefficients, weights, z)


def invert(matrix):
    """Exact inverse of a square rational matrix by Gauss-Jordan elimination."""
    m = len(matrix)
    a = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(m)] for i, row in enumerate(matrix)]
    for k in range(m):
        pivot_row = next(i for i in range(k, m) if a[i][k] != 0)
        a[k], a[pivot_row] = a[pivot_row], a[k]
        pivot = a[k][k]
        a[k] = [x / pivot for x in a[k]]
        for i in range(m):
            if i != k and a[i][k] != 0:
                factor = a[i][k]
                a[i] = [x - factor * y for x, y in zip(a[i], a[k], strict=True)]

    return [row[m:] for row in a]


def barycentric_gradients(region, cell):
    """The covectors grad lambda_o of the cell, o = 0, ..., d, as vectors G[o] in Q^r.

    For a tangent vector v, grad lambda_o . v = sum_j G[o][j] v_j is the o-th coefficient of v
    written as sum_o c_o a_o with sum_o c_o = 0, since the vertex matrix A with rows a_o is
    invertible and c = A^{-T} v.
    """
    r = region.r
    A = [[region.points[a][j] for j in range(r)] for a in cell.vertices]
    B = invert(A)
    return [[B[j][o] for j in range(r)] for o in range(r)]


def barycentric_coefficients(G, v):
    """The coefficients c with v = sum_o c_o a_o and sum_o c_o = 0 of a tangent vector v, from barycentric_gradients."""
    return [sum(G[o][j] * v[j] for j in range(len(v))) for o in range(len(G))]


def normal_and_edge_coefficients(region, cell, F, G=None):
    """The inverse of eq:reconstruction on one cell, exactly.

    F[(s, t)] for s <= t is the Bernstein coefficient vector F_{P, st} of one label, a tangent
    vector of exact rationals, and G = barycentric_gradients(region, cell) may be passed in.
    Write F_{P, st} = sum_o c_o a_o with sum_o c_o = 0.  The outward normal coefficient on the
    facet opposite o is - d V_P c_o (lem:reconstruction), and substituting these into
    eq:reconstruction leaves V_P (c_s - c_t) (a_s - a_t) / 2, so z_{P, st} = V_P (c_s - c_t) / 2.
    Returns

        outward[o][k]   the normal coefficient of the facet opposite the vertex at position o, at the
                        coefficient k of that facet, multiplied by the measure of the facet and taken
                        outward from P; the stored value is h = s_{P, f} * outward
        z[e]            z_{P, st} for the pair e = st, s < t
    """
    r, d = region.r, region.d
    V = cell.volume
    G = G or barycentric_gradients(region, cell)
    outward = [[Fraction(0)] * len(coefficient_pairs(d)) for _ in range(r)]
    z = []
    for s, t in coefficient_pairs(r):
        c = barycentric_coefficients(G, F[s, t])
        for o in range(r):
            if o not in (s, t):
                k = region.facet_pair_index[cell.facets[o]][cell.vertices[s], cell.vertices[t]]
                outward[o][k] = -d * V * c[o]
        if s < t:
            z.append(V * (c[s] - c[t]) / 2)

    return outward, z
