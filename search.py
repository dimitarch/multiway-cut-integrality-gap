"""The linear program of Section 7.1 over the family of Section 7.3, solved with HiGHS or Gurobi.

The unknowns are the unscaled data of every ordered region O_r (Section 7.2) in the
representation of Section 7.4: the normal coefficients h_{i, f, st}, the interior coefficients
z_{i, P, st} and the integrated weights V_P w_{ij, P, st}.  The program is the specialization
described in Section 7.1: the allowance is zero, the total terminal weight is normalized to
B = 1, and L is minimized, so that the floating value of the instance is 1 / L.  With
--allowance the allowance coefficients e_{P, st} >= 0 are unknowns too, the normalization is
B - R = 1, and L is minimized, which is the general program up to scaling.

Every region is of one of three types.

    free              every label has its own quadratic flow, the weights are free, and capacity is
                      the subset inequality eq:capmaxmin at every coefficient
    common-quadratic  one quadratic flow phi shared by all labels plus the constant offsets
                      A (e_i - 1/r), with every weight A / (r - 1) and capacity imposed through
                      phi_j <= A / r at every coefficient and coordinate (Section 7.1, "In higher
                      dimensions it further restricted the flows")
    common-affine     the same with phi affine, given by its normal coefficients at the facet vertices

A floating solution is written as an .npz file; rounding.py turns it into an integer table.

    python search.py 11 3 249/400 --r_0 5 --common-quadratic 6-10 --common-affine 11 --out run
    python search.py 5 2 3/5 --r_0 5 --backend gurobi --method barrier --out small
"""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import numpy as np

from geometry import TERMINAL, build_regions
from indices import coefficient_pairs, coordinate_pairs, subsets_containing_zero
from lp import GUROBI_METHODS, HIGHS_METHODS, LinearProgram, add, parse_options, solve_with_gurobi, solve_with_highs
from reconstruction import divergence_operator, numerator_operator

FREE, COMMON_QUADRATIC, COMMON_AFFINE = "free", "common-quadratic", "common-affine"


def evaluate(expression, x):
    return sum(a * x[j] for j, a in expression.items())


def amplitude_unit(r):
    """The amplitude A is stored as A / (r! (r - 1)!).

    The weights A / (r - 1) are densities on O_r, whose volume is about 1 / (r! (r - 1)!), so this
    unit keeps the objective coefficient of the amplitude near one (conditioning only).
    """
    return math.factorial(r) * math.factorial(r - 1)


def flow_expression(g, op, trace, interior, k, j):
    """V_P F_{P, st, j} = N_j(P, st) / (2 d Q) (Section 7.5) for the flow the two functions describe.

    trace(f, m) is the expression of h_{f, m}, interior(e) the expression of z_{P, e}; op is the
    numerator operator of the cell, whose integer coefficients multiply them.
    """
    scale = 1.0 / (2 * g.d * g.Q)
    expression = {}
    for slot in range(g.d):
        if op.g[k, slot, j]:
            add(expression, trace(op.facets[k, slot], op.coefficients[k, slot]), int(op.g[k, slot, j]) * scale)
    if op.z[k, j]:
        add(expression, interior(op.edges[k]), int(op.z[k, j]) * scale)
    return expression


def divergence_expression(g, op, trace, interior, s):
    """d A_{P, s} = d V_P div F_P(a_s), eq:integrateddiv, for the flow the two functions describe."""
    expression = {}
    for slot in range(g.d):
        for m in range(g.d):
            add(expression, trace(op.facets[s, slot, m], op.coefficients[s, slot, m]), int(op.weights[s, slot, m]))
    for e in range(len(coordinate_pairs(g.r))):
        if op.z[s, e]:
            add(expression, interior(e), int(op.z[s, e]))
    return expression


class FreeRegion:
    """Unknowns h[i, f, k], z[c, i, e] and V_P w as W[c, k, e] >= 0 (Section 7.4)."""

    kind = FREE

    def __init__(self, lp, region):
        self.g = region
        r = region.r
        self.h = lp.variables((r, len(region.facets), len(coefficient_pairs(region.d))))
        self.z = lp.variables((len(region.cells), r, len(coordinate_pairs(r))))
        self.W = lp.variables((len(region.cells), len(coefficient_pairs(r)), len(coordinate_pairs(r))), lower=0.0)
        self.numerators = [numerator_operator(region, cell) for cell in region.cells]
        self.divergences = [divergence_operator(region, cell) for cell in region.cells]

    def trace(self, i, f, k):
        """h_{i, f, st} for the coefficient k = st of facet f."""
        return {int(self.h[i, f, k]): 1.0}

    def flow(self, c, i, k, j):
        """V_P F_{i, P, st, j} for the coefficient k = st."""
        return flow_expression(self.g, self.numerators[c], lambda f, m: {int(self.h[i, f, m]): 1.0}, lambda e: {int(self.z[c, i, e]): 1.0}, k, j)

    def vertex_divergence(self, c, i, s):
        """d A_{P, i, s} = d V_P div F_{i, P}(a_s), eq:integrateddiv."""
        return divergence_expression(self.g, self.divergences[c], lambda f, m: {int(self.h[i, f, m]): 1.0}, lambda e: {int(self.z[c, i, e]): 1.0}, s)

    def integrated_weight(self, c, k, e):
        return {int(self.W[c, k, e]): 1.0}

    def add_capacity(self, lp):
        """eq:capmaxmin at every coefficient: for every subset I with 0 in I,
        max(0, max_l f_l(I)) - min(0, min_l f_l(I)) <= w(I), with f_l(I) = sum_{j in I} V_P F_{l, P, st, j}.

        One unknown m <= 0 per subset stands for min(0, min_l f_l(I)): the rows are m <= f_l(I),
        f_l(I) - m <= w(I) for every label l, and -m <= w(I) for the label *.
        """
        g = self.g
        r = g.r
        masks = subsets_containing_zero(r)
        pairs = coordinate_pairs(r)
        for c in range(len(g.cells)):
            for k in range(len(coefficient_pairs(r))):
                flows = [[self.flow(c, i, k, j) for j in range(r)] for i in range(r)]
                for mask in masks:
                    inside = np.flatnonzero(mask)
                    weight = {int(self.W[c, k, e]): 1.0 for e, (a, b) in enumerate(pairs) if mask[a] != mask[b]}
                    m = lp.variable(upper=0.0)
                    lp.at_most(add({m: -1.0}, weight, -1.0))
                    for i in range(r):
                        f = {}
                        for j in inside:
                            add(f, flows[i][j])
                        lp.at_most(add({m: 1.0}, f, -1.0))
                        lp.at_most(add(add({m: -1.0}, f), weight, -1.0))

    def arrays(self, x):
        r = self.g.r
        return {f"h_{r}": x[self.h], f"z_{r}": x[self.z], f"W_{r}": x[self.W]}

    def fix(self, lp, stored):
        r = self.g.r
        for ids, key in ((self.h, f"h_{r}"), (self.z, f"z_{r}"), (self.W, f"W_{r}")):
            for j, value in zip(ids.flat, stored[key].flat, strict=True):
                lp.fix(int(j), value)

    def expand(self, x):
        """The floating h[i, f, k], z[c, i, e] and V_P w[c, k, e] arrays of the full representation."""
        return x[self.h], x[self.z], x[self.W]


class CommonQuadraticRegion:
    """One quadratic flow phi, unknown through phi_h[f, k] and phi_z[c, e], plus the offsets A (e_i - 1/r).

    The label-i flow is phi + A (e_i - 1/r).  Its normal coefficient on facet f is
    phi_h[f, k] + A c[f, i], where c[f, i] is the normal coefficient of the constant field
    e_i - 1/r, and its interior coefficient is phi_z[c, e] + A zeta[c, e, i].  Every weight is
    A / (r - 1), so the weight crossing a subset I is at least A, and phi_j <= A / r at every
    coefficient gives capacity: the subset sums of the labels and of * span exactly A.

    For the interior point method, phi_h is not a free unknown but phi_h[f, k] = W_f z[f, k] + L_f A
    with 0 <= z <= A, where W_f = max_i c[f, i] - min_i c[f, i] and L_f = - max_i c[f, i], so that
    the label traces phi_h + A c[f, i] range over [A (c[f, i] - max c), A (c[f, i] - min c)].
    The amplitude A is kept in the unit amplitude_unit(r).
    """

    kind = COMMON_QUADRATIC

    def __init__(self, lp, region):
        self.g = region
        r = region.r
        self.unit = amplitude_unit(r)
        self.A = lp.variable(lower=0.0)
        self.z = lp.variables((len(region.facets), len(coefficient_pairs(region.d))), lower=0.0)
        self.phi_z = lp.variables((len(region.cells), len(coordinate_pairs(r))))
        self.numerators = [numerator_operator(region, cell) for cell in region.cells]
        self.divergences = [divergence_operator(region, cell) for cell in region.cells]
        self.c, self.zeta = offsets(region)
        self.scaling = [(self.unit * (max(c) - min(c)), -self.unit * max(c)) for c in self.c]
        for column in self.z.flat:
            lp.at_most({int(column): 1.0, self.A: -1.0})

    def amplitude(self, factor=1.0):
        return {self.A: self.unit * factor}

    def phi_h(self, f, k):
        W, L = self.scaling[f]
        return {int(self.z[f, k]): W, self.A: L}

    def trace(self, i, f, k):
        return add(self.phi_h(f, k), self.amplitude(self.c[f][i]))

    def common_flow(self, c, k, j):
        """V_P phi_{P, st, j} for the coefficient k = st."""
        return flow_expression(self.g, self.numerators[c], self.phi_h, lambda e: {int(self.phi_z[c, e]): 1.0}, k, j)

    def vertex_divergence(self, c, i, s):
        """d V_P div phi(a_s); the constant offsets have zero divergence, so this is the same for every label."""
        return divergence_expression(self.g, self.divergences[c], self.phi_h, lambda e: {int(self.phi_z[c, e]): 1.0}, s)

    def integrated_weight(self, c, k, e):
        return self.amplitude(float(self.g.cells[c].volume) / self.g.d)

    def add_capacity(self, lp):
        """V_P phi_{st, j} <= V_P A / r for every coefficient st and coordinate j."""
        g = self.g
        for c, cell in enumerate(g.cells):
            for k in range(len(coefficient_pairs(g.r))):
                for j in range(g.r):
                    lp.at_most(add(self.common_flow(c, k, j), self.amplitude(-float(cell.volume) / g.r)))

    def arrays(self, x):
        r = self.g.r
        phi_h = np.array([[evaluate(self.phi_h(f, k), x) for k in range(self.z.shape[1])] for f in range(self.z.shape[0])])
        return {f"phi_h_{r}": phi_h, f"phi_z_{r}": x[self.phi_z], f"A_{r}": np.array(self.unit * x[self.A])}

    def fix(self, lp, stored):
        r = self.g.r
        amplitude = float(stored[f"A_{r}"]) / self.unit
        lp.fix(self.A, amplitude)
        for f, (W, L) in enumerate(self.scaling):
            for column, phi_h in zip(self.z[f], stored[f"phi_h_{r}"][f], strict=True):
                lp.fix(int(column), min(max((phi_h - amplitude * L) / W, 0.0), amplitude))
        for column, value in zip(self.phi_z.flat, stored[f"phi_z_{r}"].flat, strict=True):
            lp.fix(int(column), value)

    def expand(self, x):
        g = self.g
        r = g.r
        A = self.unit * x[self.A]
        phi_h = self.arrays(x)[f"phi_h_{r}"]
        h = phi_h[None, :, :] + A * np.array(self.c).T[:, :, None]
        z = x[self.phi_z][:, None, :] + A * np.array(self.zeta).transpose(0, 2, 1)
        W = np.array([[[A * float(cell.volume) / g.d] * len(coordinate_pairs(r))] * len(coefficient_pairs(r)) for cell in g.cells])
        return h, z, W


class CommonAffineRegion:
    """One affine flow phi, unknown through its normal coefficients phi_v[f, v] at the vertices v of every facet f.

    The quadratic normal coefficient at st is (phi_v[f, s] + phi_v[f, t]) / 2 (Section 7.5,
    averaging rule).  The value of phi at a vertex a_v of a cell is determined by the d normal
    coefficients at a_v: phi(a_v) = sum_o c_o a_o with c_o = - s_{P, f_o} phi_v[f_o, v] / (d V_P)
    for o != v and c_v = - sum_{o != v} c_o.  The offsets A (e_i - 1/r) and the weights are as in
    CommonQuadraticRegion.
    """

    kind = COMMON_AFFINE

    def __init__(self, lp, region):
        self.g = region
        r = region.r
        self.unit = amplitude_unit(r)
        self.A = lp.variable(lower=0.0)
        self.z = lp.variables((len(region.facets), region.d), lower=0.0)
        self.c, self.zeta = offsets(region)
        # phi_v[f, v] = W_f z[f, v] + L_f A with z >= 0, W_f = max c - min c and L_f = - max c as in CommonQuadraticRegion.
        self.scaling = [(self.unit * (max(c) - min(c)), -self.unit * max(c)) for c in self.c]
        self.vertex_position = [{a: v for v, a in enumerate(facet)} for facet in region.facets]
        self.z_of_phi = [self._interior_coefficients(c) for c in range(len(region.cells))]

    def amplitude(self, factor=1.0):
        return {self.A: self.unit * factor}

    def vertex_trace(self, f, a):
        """phi_v at the vertex a of facet f, outward from the first incident cell."""
        W, L = self.scaling[f]
        return {int(self.z[f, self.vertex_position[f][a]]): W, self.A: L}

    def trace(self, i, f, k):
        s, t = coefficient_pairs(self.g.d)[k]
        facet = self.g.facets[f]
        expression = add(add({}, self.vertex_trace(f, facet[s]), 0.5), self.vertex_trace(f, facet[t]), 0.5)
        return add(expression, self.amplitude(self.c[f][i]))

    def value(self, c, v, j):
        """V_P phi(a_v)_j = sum_{o != v} (- s_{P, f_o} phi_v[f_o, v] / d) (a_{o, j} - a_{v, j})."""
        g = self.g
        cell = g.cells[c]
        a_v = g.points[cell.vertices[v]]
        expression = {}
        for o in range(g.r):
            if o != v:
                a_o = g.points[cell.vertices[o]]
                add(expression, self.vertex_trace(cell.facets[o], cell.vertices[v]), -cell.signs[o] * float(a_o[j] - a_v[j]) / g.d)
        return expression

    def _interior_coefficients(self, c):
        """z_{P, st} of the affine flow as linear expressions in phi_v.

        With phi(a_v) = sum_o c^v_o a_o as in the class docstring, the coefficient vector at st
        is (phi(a_s) + phi(a_t)) / 2, so z_{P, st} = V_P ((c^s_s + c^t_s) - (c^s_t + c^t_t)) / 4
        (normal_and_edge_coefficients).
        """
        g = self.g
        r, d = g.r, g.d
        cell = g.cells[c]
        V = float(cell.volume)

        def coefficient(v, o):
            """c^v_o as a linear expression in phi_v."""
            if o != v:
                return add({}, self.vertex_trace(cell.facets[o], cell.vertices[v]), -cell.signs[o] / (d * V))
            expression = {}
            for u in range(r):
                if u != v:
                    add(expression, coefficient(v, u), -1.0)
            return expression

        expressions = []
        for s, t in coordinate_pairs(r):
            z = {}
            for v in (s, t):
                add(z, coefficient(v, s), V / 4)
                add(z, coefficient(v, t), -V / 4)
            expressions.append(z)
        return expressions

    def vertex_divergence(self, c, i, s):
        """d V_P div phi = sum_f s_{P, f} sum_{v in f} phi_v[f, v], constant on the cell and the same for every label."""
        cell = self.g.cells[c]
        expression = {}
        for o in range(self.g.r):
            for a in self.g.facets[cell.facets[o]]:
                add(expression, self.vertex_trace(cell.facets[o], a), cell.signs[o])
        return expression

    def integrated_weight(self, c, k, e):
        return self.amplitude(float(self.g.cells[c].volume) / self.g.d)

    def add_capacity(self, lp):
        """V_P phi(a_v)_j <= V_P A / r at every vertex and coordinate; the coefficients are averages of vertex values."""
        g = self.g
        for c, cell in enumerate(g.cells):
            for v in range(g.r):
                for j in range(g.r):
                    lp.at_most(add(self.value(c, v, j), self.amplitude(-float(cell.volume) / g.r)))

    def arrays(self, x):
        r = self.g.r
        phi_v = np.array([[evaluate(self.vertex_trace(f, a), x) for a in facet] for f, facet in enumerate(self.g.facets)])
        return {f"phi_v_{r}": phi_v, f"A_{r}": np.array(self.unit * x[self.A])}

    def fix(self, lp, stored):
        r = self.g.r
        amplitude = float(stored[f"A_{r}"]) / self.unit
        lp.fix(self.A, amplitude)
        for f, (W, L) in enumerate(self.scaling):
            for column, phi_v in zip(self.z[f], stored[f"phi_v_{r}"][f], strict=True):
                lp.fix(int(column), max((phi_v - amplitude * L) / W, 0.0))

    def expand(self, x):
        g = self.g
        r = g.r
        A = self.unit * x[self.A]
        pairs = coefficient_pairs(g.d)
        phi_v = self.arrays(x)[f"phi_v_{r}"]
        phi_h = np.array([[(phi_v[f, s] + phi_v[f, t]) / 2 for s, t in pairs] for f in range(len(g.facets))])
        h = phi_h[None, :, :] + A * np.array(self.c).T[:, :, None]
        phi_z = np.array([[evaluate(expression, x) for expression in self.z_of_phi[c]] for c in range(len(g.cells))])
        z = phi_z[:, None, :] + A * np.array(self.zeta).transpose(0, 2, 1)
        W = np.array([[[A * float(cell.volume) / g.d] * len(coordinate_pairs(r))] * len(coefficient_pairs(r)) for cell in g.cells])
        return h, z, W


def offsets(region):
    """The representation of the constant fields e_i - 1/r on every cell, by the inverse of eq:reconstruction.

    With e_i - 1/r = sum_o c_o a_o, the outward normal coefficient on the facet opposite o is
    - d V_P c_o and the interior coefficient at st is V_P (c_s - c_t) / 2 (normal_and_edge_coefficients).
    Floating point, since only the program uses these.

    c[f][i]        normal coefficient of e_i - 1/r on facet f, outward from its first incident cell
    zeta[c][e][i]  interior coefficient z of e_i - 1/r on cell c at the pair e
    """
    r, d = region.r, region.d
    c = [[None] * r for _ in region.facets]
    zeta = []
    for cell in region.cells:
        A = np.array([[float(x) for x in region.points[a]] for a in cell.vertices])
        G = np.linalg.inv(A).T  # row o: grad lambda_o, so that G @ v are the coefficients c of a tangent v
        coefficients = G @ (np.eye(r) - 1.0 / r)  # column i: the coefficients of e_i - 1/r
        V = float(cell.volume)
        zeta.append([[V * (coefficients[s, i] - coefficients[t, i]) / 2 for i in range(r)] for s, t in coordinate_pairs(r)])
        for o in range(r):
            if cell.signs[o] == 1:
                c[cell.facets[o]] = [-d * V * coefficients[o, i] for i in range(r)]

    return c, zeta


REGION_TYPES = {FREE: FreeRegion, COMMON_QUADRATIC: CommonQuadraticRegion, COMMON_AFFINE: CommonAffineRegion}


def add_boundary_rules(lp, model):
    """The two boundary rules of Section 7.4: bounds on the unknowns of a free region, rows on the traces otherwise.

    In a common-affine region the traces are affine on every facet, so the rules are imposed at the
    vertices of the facet; the quadratic coefficients are their averages.
    """
    g = model.g
    for f, kind in g.boundary.items():
        if kind == TERMINAL:
            zero, opposite = range(1, g.r), None
        elif isinstance(kind, int):
            zero, opposite = [i for i in range(g.r) if i not in (kind, kind + 1)], kind
        else:
            continue

        if model.kind == FREE:
            for k in range(len(coefficient_pairs(g.d))):
                for i in zero:
                    lp.fix(int(model.h[i, f, k]), 0.0)
                if opposite is None:
                    lp.lower[int(model.h[0, f, k])] = 0.0
                else:
                    lp.equal({int(model.h[opposite, f, k]): 1.0, int(model.h[opposite + 1, f, k]): 1.0})
            continue

        if model.kind == COMMON_AFFINE:
            # The offsets c[f, i] agree for all labels in `zero`, so one row at each vertex fixes phi_v
            # and the remaining rows, including label 0's nonnegativity, are implied.
            for a in g.facets[f]:
                lp.equal(add(model.vertex_trace(f, a), model.amplitude(model.c[f][zero[0]])))
            continue

        traces = [[model.trace(i, f, k) for i in range(g.r)] for k in range(len(coefficient_pairs(g.d)))]
        for trace in traces:
            for i in zero:
                lp.equal(trace[i])
            if opposite is None:
                lp.at_most({j: -a for j, a in trace[0].items()})
            else:
                lp.equal(add(trace[opposite], trace[opposite + 1]))


def add_conservation(lp, model, parent, allowance):
    """(A_{P, i, s} + A_{P, i, t}) / 2 <= h_{i, f^+, st} + e_{P, st} at every coefficient of every cell (eq:residual).

    `allowance` is None for the zero-allowance program, else the array of allowance unknowns e[c, k].
    Rows that other rows imply are left out, as in the earlier implementation: in a common region every
    label has the divergence of phi and a common parent supplies every label of the child equally, so
    one label suffices; the divergence is affine on the cell, so when the parent's supply is affine too
    (common-affine parent) the row at st is the average of the rows at ss and tt; and in a common-affine
    region without parent the divergence is constant, so one row per cell suffices.
    """
    g = model.g
    r, d = g.r, g.d
    pairs = coefficient_pairs(r)
    labels = [0] if model.kind != FREE and (parent is None or parent.kind != FREE) else range(r)
    if parent is not None and parent.kind == COMMON_AFFINE:
        rows = [k for k, (s, t) in enumerate(pairs) if s == t]
    elif parent is None and model.kind == COMMON_AFFINE:
        rows = [0]
    else:
        rows = range(len(pairs))

    for c, cell in enumerate(g.cells):
        if parent is not None:
            f_plus, vertex_map = g.parent_facet(parent.g, cell)
        for i in labels:
            divergence = [model.vertex_divergence(c, i, s) for s in range(r)]
            for k in rows:
                s, t = pairs[k]
                row = add(add({}, divergence[s], 0.5 / d), divergence[t], 0.5 / d)
                if parent is not None:
                    k_plus = parent.g.facet_pair_index[f_plus][tuple(sorted((vertex_map[cell.vertices[s]], vertex_map[cell.vertices[t]])))]
                    add(row, parent.trace(i, f_plus, k_plus), -1.0)
                if allowance is not None:
                    add(row, {int(allowance[c, k]): -1.0})
                lp.at_most(row)


def build(q, T, n, r_0, types, allowance=False):
    """The program for the regions O_2, ..., O_q of the given types; returns (lp, models, normalization row)."""
    regions = build_regions(q, T, n, r_0)
    lp = LinearProgram()
    models = {r: REGION_TYPES[types[r]](lp, g) for r, g in regions.items()}
    normalization = {}  # B - R, fixed to 1
    for r, model in models.items():
        g = model.g
        terminal_weight = {}
        for f, kind in g.boundary.items():
            if kind == TERMINAL:
                for k in range(len(coefficient_pairs(g.d))):
                    add(terminal_weight, model.trace(0, f, k), 2 / (r * (r - 1)))
        add(normalization, terminal_weight)
        add(lp.objective, terminal_weight, float(1 - T))
        for c in range(len(g.cells)):
            for k in range(len(coefficient_pairs(r))):
                for e in range(len(coordinate_pairs(r))):
                    add(lp.objective, model.integrated_weight(c, k, e), 2 / (r * (r + 1)))

        allowances = None
        if allowance:
            allowances = lp.variables((len(g.cells), len(coefficient_pairs(r))), lower=0.0)
            for j in allowances.flat:
                add(normalization, {int(j): -2 / (r * (r + 1))})
        add_boundary_rules(lp, model)
        add_conservation(lp, model, models.get(r + 1), allowances)
        model.add_capacity(lp)
    lp.equal(normalization, 1.0)
    return lp, models


def parse_regions(text, q):
    """'6-10,11' -> [6, 7, 8, 9, 10, 11], every entry between 2 and q."""
    regions = set()
    for part in filter(None, (text or "").split(",")):
        first, _, last = part.partition("-")
        regions.update(range(int(first), int(last or first) + 1))
    assert all(2 <= r <= q for r in regions), f"regions must lie between 2 and {q}"
    return sorted(regions)


def region_types(q, common_quadratic, common_affine):
    types = {r: FREE for r in range(2, q + 1)}
    for kind, text in ((COMMON_QUADRATIC, common_quadratic), (COMMON_AFFINE, common_affine)):
        for r in parse_regions(text, q):
            types[r] = kind
    return types


def fix_regions(lp, models, path, fixed, q, n, r_0, T):
    """Hold the given regions at the values of a stored floating solution of the same family and region types."""
    stored = np.load(path)
    same = (int(stored["n"]), int(stored["r_0"]), Fraction(int(stored["T_num"]), int(stored["T_den"]))) == (n, r_0, T)
    assert same and int(stored["q"]) >= max(fixed), f"{path} belongs to a different family (n, r_0, T) or has too few regions"
    stored_types = json.loads(str(stored["types"]))
    for r in fixed:
        assert stored_types[str(r)] == models[r].kind, f"region {r} is {models[r].kind} here but {stored_types[str(r)]} in {path}"
        models[r].fix(lp, stored)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("q", type=int, help="number of coordinates of the top face")
    parser.add_argument("n", type=int, help="subdivision parameter")
    parser.add_argument("T", type=Fraction, help="threshold, a rational in (1/2, 1), e.g. 249/400")
    parser.add_argument("--r_0", type=int, help="regions up to r_0 are subdivided by eq:subdivision, higher ones by cones (default q)")
    parser.add_argument("--common-quadratic", help="regions of type common-quadratic, e.g. 6-10 or 6,7,8")
    parser.add_argument("--common-affine", help="regions of type common-affine, e.g. 11")
    parser.add_argument("--allowance", action="store_true", help="allowance unknowns e_{P, st} >= 0 and normalization B - R = 1")
    parser.add_argument("--fix-from", help="floating solution (.npz) whose regions --fix are held fixed")
    parser.add_argument("--fix", help="regions to hold fixed, e.g. 2-5,10-11")
    parser.add_argument("--backend", default="highs", choices=["highs", "gurobi"])
    parser.add_argument(
        "--method", default="auto", help=f"HiGHS: {', '.join(HIGHS_METHODS)} (auto: ipm); Gurobi: {', '.join(GUROBI_METHODS)} (auto: barrier)"
    )
    parser.add_argument("--threads", type=int, default=0, help="solver threads (0: the solver decides)")
    parser.add_argument("--time-limit", type=float, default=math.inf, help="seconds")
    parser.add_argument("--feasibility", type=float, default=1e-9, help="primal feasibility tolerance")
    parser.add_argument("--optimality", type=float, default=1e-7, help="relative optimality tolerance")
    parser.add_argument("--crossover", action="store_true", help="run crossover to a basic solution after the interior point method")
    parser.add_argument("--option", action="append", metavar="NAME=VALUE", help="extra solver option or parameter, repeatable")
    parser.add_argument("--out", required=True, help="output stem: writes <out>.npz, <out>.json and <out>.log")
    parser.add_argument("--build-only", action="store_true", help="report the size of the program and stop")
    args = parser.parse_args()

    r_0 = args.r_0 or args.q
    types = region_types(args.q, args.common_quadratic, args.common_affine)
    start = time.monotonic()
    lp, models = build(args.q, args.T, args.n, r_0, types, args.allowance)
    fixed = parse_regions(args.fix, args.q)
    if fixed:
        assert args.fix_from, "--fix needs --fix-from"
        fix_regions(lp, models, args.fix_from, fixed, args.q, args.n, r_0, args.T)

    method = args.method
    if method == "auto":
        method = "barrier" if args.backend == "gurobi" else "ipm"
    allowed = GUROBI_METHODS if args.backend == "gurobi" else HIGHS_METHODS
    assert method in allowed, f"--method {method} is not one of {allowed}"
    info = dict(
        q=args.q,
        n=args.n,
        T=str(args.T),
        r_0=r_0,
        types=types,
        allowance=args.allowance,
        fixed=fixed,
        fix_from=args.fix_from,
        columns=lp.num_columns,
        rows=lp.num_rows,
        nonzeros=lp.num_nonzeros,
        build_seconds=round(time.monotonic() - start, 1),
        backend=args.backend,
        method=method,
        threads=args.threads,
        feasibility=args.feasibility,
        optimality=args.optimality,
        crossover=args.crossover,
        options=parse_options(args.option),
    )
    print(json.dumps(info), flush=True)
    if args.build_only:
        return

    solve = solve_with_gurobi if args.backend == "gurobi" else solve_with_highs
    x, result = solve(
        lp,
        method,
        args.threads,
        args.time_limit,
        args.feasibility,
        args.optimality,
        args.crossover,
        log_file=args.out + ".log",
        options=parse_options(args.option),
    )
    info.update(result, seconds=round(time.monotonic() - start, 1))
    if x is not None:
        L = float(lp.cost_vector() @ x)
        info.update(L=L, value=1 / L, violation=lp.violation(x), solution=args.out + ".npz")
        arrays = dict(q=args.q, n=args.n, r_0=r_0, T_num=args.T.numerator, T_den=args.T.denominator, types=json.dumps(types))
        for model in models.values():
            arrays.update(model.arrays(x))
        np.savez_compressed(args.out + ".npz", **arrays)
    Path(args.out + ".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({key: value for key, value in info.items() if key != "types"}), flush=True)


if __name__ == "__main__":
    main()
