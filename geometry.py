"""The subdivision of the ordered regions O_r into simplex cells (Section 7.2) and the incidence
data that Section 7.4 reads off it: the facets of every cell, their orientation s_{P, f}, the
opposite vertices o_{P, f}, the volumes V_P, and the boundary kind of every facet of O_r.

Every coordinate is an exact rational.  The vertices of O_r are also stored as integer
numerators p_{s, j} over the common denominator Q_r of Section 7.5.
"""

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math

import numpy as np

from indices import coefficient_pairs

TERMINAL = "terminal"  # the facet lies on x_0 = T
COORDINATE = "coordinate"  # the facet lies on x_{r - 1} = 0, the copy of O_{r - 1}
# an integer j means the facet lies on the equality facet x_j = x_{j + 1} of O_r


def region_vertices(r, T):
    """The vertices u_2, ..., u_r and v_2, ..., v_r of O_r (Section 7.2), as dicts indexed by j."""
    u = {j: tuple(Fraction(1, j) if i < j else Fraction(0) for i in range(r)) for j in range(2, r + 1)}
    v = {j: (T,) + tuple((1 - T) / (j - 1) if i < j else Fraction(0) for i in range(1, r)) for j in range(2, r + 1)}
    return u, v


def initial_simplices(r, T):
    """The simplices conv(u_2, ..., u_s, v_s, ..., v_r), 2 <= s <= r, of eq:initialmesh, in that vertex order."""
    u, v = region_vertices(r, T)
    return [tuple(u[j] for j in range(2, s + 1)) + tuple(v[j] for j in range(s, r + 1)) for s in range(2, r + 1)]


def kuhn_paths(n, d):
    """The lattice paths z, z + e_sigma(1), ..., z + (1, ..., 1) of Section 7.2 whose vectors are all weakly increasing."""
    paths = []
    for z in itertools.combinations_with_replacement(range(n), d):
        for sigma in itertools.permutations(range(d)):
            y = list(z)
            path = [tuple(y)]
            for j in sigma:
                y[j] += 1
                if any(y[h] > y[h + 1] for h in range(d - 1)):
                    break
                path.append(tuple(y))
            if len(path) == d + 1:
                paths.append(path)

    assert len(paths) == n**d
    return paths


def kuhn_subdivision(simplex, n):
    """Subdivide a simplex with ordered vertices into n^d simplices, one per path, by eq:subdivision."""
    d = len(simplex) - 1
    r = len(simplex[0])
    cells = []
    for path in kuhn_paths(n, d):
        cell = []
        for y in path:
            weights = [Fraction(n - y[-1], n), Fraction(y[0], n)] + [Fraction(y[k] - y[k - 1], n) for k in range(1, d)]
            cell.append(tuple(sum(weights[s] * simplex[s][j] for s in range(d + 1)) for j in range(r)))
        cells.append(frozenset(cell))

    return cells


def subdivision_cells(r, T, n, r_0, child=None):
    """The cells of O_r as sets of rational points.

    For r <= r_0 every initial simplex is subdivided into n^(r - 1) cells.  For r > r_0 every cell
    of the child region O_{r - 1}, with a zero coordinate appended, is joined to v_r, and every
    facet of the child subdivision on x_0 = x_1 is joined to u_r and v_r (Section 7.2).
    """
    if r <= r_0:
        return [cell for simplex in initial_simplices(r, T) for cell in kuhn_subdivision(simplex, n)]

    assert child is not None and child.r == r - 1
    u, v = region_vertices(r, T)
    lift = {a: point + (Fraction(0),) for a, point in enumerate(child.points)}
    cells = [frozenset(lift[a] for a in cell.vertices) | {v[r]} for cell in child.cells]
    for f, kind in child.boundary.items():
        if kind == 0:
            cells.append(frozenset(lift[a] for a in child.facets[f]) | {u[r], v[r]})

    return cells


def determinant(rows):
    """Exact determinant of a square integer matrix (Bareiss elimination); 1 for the empty matrix."""
    a = [[int(x) for x in row] for row in rows]
    m = len(a)
    sign = 1
    previous = 1
    for k in range(m - 1):
        if a[k][k] == 0:
            pivot_row = next((i for i in range(k + 1, m) if a[i][k] != 0), None)
            if pivot_row is None:
                return 0
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign = -sign
        for i in range(k + 1, m):
            for j in range(k + 1, m):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // previous
        previous = a[k][k]

    return sign * a[-1][-1] if m else 1


@dataclass(frozen=True)
class Cell:
    """A simplex cell P of the subdivision of O_r with the incidence data of Section 7.4."""

    vertices: tuple  # vertex indices a_0 < a_1 < ... < a_d
    facets: tuple  # facets[o] is the index of the facet of P opposite the vertex at position o
    signs: tuple  # signs[o] = s_{P, f} for that facet: +1 if P is its first incident cell, else -1
    volume: Fraction  # V_P


class Region:
    """The subdivision of one ordered region O_r, indexed as stored in a table (README.md, table format).

    points   the vertices, exact rationals, and the integers p[s, j] with points[s][j] = p[s, j] / Q
    cells    sorted vertex-index tuples, in table order
    facets   sorted vertex-index tuples, in table order; facet f is oriented outward from its first
             incident cell facet_cells[f][0]
    boundary facet index -> TERMINAL, COORDINATE, or j for the equality facet x_j = x_{j + 1}, for
             the facets of O_r itself; interior facets of O_r are absent
    """

    def __init__(self, r, T, points, cells, facets=None):
        self.r, self.d, self.T = r, r - 1, Fraction(T)
        self.points = [tuple(Fraction(x) for x in point) for point in points]
        self.Q = math.lcm(*(x.denominator for point in self.points for x in point))
        self.p = np.array([[int(x * self.Q) for x in point] for point in self.points], dtype=np.int64).reshape(-1, r)
        self.point_index = {point: a for a, point in enumerate(self.points)}
        self._check_points()

        cells = [tuple(cell) for cell in cells]
        assert all(len(cell) == r and list(cell) == sorted(set(cell)) for cell in cells), "cells must be increasing vertex tuples"
        assert len(set(cells)) == len(cells), "duplicate cell"
        assert {a for cell in cells for a in cell} == set(range(len(self.points))), "every vertex must lie in a cell"
        self.facets, self.facet_index = self._index_facets(cells, facets)
        self.facet_cells = [[] for _ in self.facets]
        self.cells = []
        for c, cell in enumerate(cells):
            cell_facets = tuple(self.facet_index[cell[:o] + cell[o + 1 :]] for o in range(r))
            signs = []
            for o, f in enumerate(cell_facets):
                signs.append(1 if not self.facet_cells[f] else -1)
                self.facet_cells[f].append((c, o))
            self.cells.append(Cell(cell, cell_facets, tuple(signs), self._volume(cell)))

        assert all(1 <= len(incident) <= 2 for incident in self.facet_cells), "a facet belongs to one or two cells"
        self.boundary = {f: self._boundary_kind(f) for f, incident in enumerate(self.facet_cells) if len(incident) == 1}
        self._check_interior_facets()
        self._check_measures()

        # Coefficient index within facet f of a pair of its vertices (Section 7.2, pairs st of cell vertices).
        self.facet_pair_index = [{(facet[s], facet[t]): k for k, (s, t) in enumerate(coefficient_pairs(self.d))} for facet in self.facets]
        # The coordinate facets by their point sets, with the zero coordinate removed, for matching with the child's cells.
        self.coordinate_facets = {
            frozenset(self.points[a][:-1] for a in self.facets[f]): f for f, kind in self.boundary.items() if kind == COORDINATE
        }

    @classmethod
    def canonical(cls, r, T, n, r_0, child=None):
        """Build the subdivision of Section 7.2 with the indexing a table receives when it is created here."""
        points, index, cells = [], {}, []
        for cell in subdivision_cells(r, T, n, r_0, child):
            for point in sorted(cell):
                if point not in index:
                    index[point] = len(points)
                    points.append(point)
            cells.append(tuple(sorted(index[point] for point in cell)))

        region = cls(r, T, points, cells)
        if child is not None:
            region.check_child(child)
        return region

    def check_child(self, child):
        """The restriction of the subdivision to the facet x_{r - 1} = 0 is the child's subdivision (lem:symmetry)."""
        child_cells = {frozenset(child.points[a] for a in cell.vertices) for cell in child.cells}
        assert set(self.coordinate_facets) == child_cells, "the coordinate facet is not subdivided like O_{r - 1}"

    def cell_point_sets(self):
        return {frozenset(self.points[a] for a in cell.vertices) for cell in self.cells}

    def parent_facet(self, parent, cell):
        """The coordinate facet f^+ of the parent region matching this cell, and the parent vertex of each cell vertex."""
        f = parent.coordinate_facets[frozenset(self.points[a] for a in cell.vertices)]
        vertex_map = {a: parent.point_index[self.points[a] + (Fraction(0),)] for a in cell.vertices}
        return f, vertex_map

    # ---- construction details ----

    def _index_facets(self, cells, facets):
        expected = []
        seen = set()
        for cell in cells:
            for o in range(self.r):
                facet = cell[:o] + cell[o + 1 :]
                if facet not in seen:
                    seen.add(facet)
                    expected.append(facet)

        if facets is None:
            facets = expected
        facets = [tuple(facet) for facet in facets]
        assert sorted(facets) == sorted(expected), "the facet list must be exactly the facets of the cells"
        return facets, {facet: f for f, facet in enumerate(facets)}

    def _volume(self, cell):
        """V_P = |det(a_0, ..., a_d)| / d! in the volume measure of Section 5.1, from the integer points."""
        return Fraction(abs(determinant([self.p[a] for a in cell])), self.Q**self.r * math.factorial(self.d))

    def _boundary_kind(self, f):
        points = [self.points[a] for a in self.facets[f]]
        kinds = []
        if all(x[0] == self.T for x in points):
            kinds.append(TERMINAL)
        if all(x[-1] == 0 for x in points):
            kinds.append(COORDINATE)
        kinds += [j for j in range(self.r - 1) if all(x[j] == x[j + 1] for x in points)]

        assert len(kinds) == 1, f"facet {f} of O_{self.r} belongs to one cell but lies on {len(kinds)} boundary hyperplanes"
        return kinds[0]

    def _check_points(self):
        for point in self.points:
            assert len(point) == self.r and sum(point) == 1, "a vertex must lie in the simplex"
            assert all(point[j] >= point[j + 1] for j in range(self.d)) and point[-1] >= 0 and point[0] <= self.T, "a vertex must lie in O_r"

    def _check_interior_facets(self):
        """Two cells sharing a facet lie on opposite sides of it."""
        for f, incident in enumerate(self.facet_cells):
            if len(incident) == 2:
                orientations = [determinant([self.p[a] for a in self.facets[f]] + [self.p[self.cells[c].vertices[o]]]) for c, o in incident]
                assert orientations[0] * orientations[1] < 0, f"cells {incident} overlap across facet {f}"

    def _check_measures(self):
        """The volume of O_r and the measure of its terminal facet, eq:volumes."""
        r, d, T, Q = self.r, self.d, self.T, self.Q
        volume = sum(cell.volume for cell in self.cells)
        assert volume == (1 - r * (1 - T) ** d) / (math.factorial(r) * math.factorial(d)), "cell volumes do not add up to |O_r|"

        terminal = Fraction(0)
        for f, kind in self.boundary.items():
            if kind == TERMINAL:
                # Surface measure on x_0 = T: Lebesgue measure after deleting x_0 and one more coordinate (Section 5.1).
                p = [self.p[a] for a in self.facets[f]]
                terminal += Fraction(
                    abs(determinant([[p[k][j] - p[0][j] for j in range(2, r)] for k in range(1, d)])), Q ** (d - 1) * math.factorial(d - 1)
                )
        assert terminal == (1 - T) ** (r - 2) / (math.factorial(r - 2) * math.factorial(d)), "terminal facets do not add up"


def build_regions(q, T, n, r_0):
    """The canonical subdivisions of O_2, ..., O_q."""
    regions = {}
    for r in range(2, q + 1):
        regions[r] = Region.canonical(r, T, n, r_0, regions.get(r - 1))
    return regions
