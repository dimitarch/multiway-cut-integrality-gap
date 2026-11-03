"""The subdivision of Section 7.2: cell counts, the facet restriction, the boundary facets."""

from fractions import Fraction

import pytest

from geometry import COORDINATE, TERMINAL, build_regions, kuhn_paths, kuhn_subdivision

T = Fraction(249, 400)


def test_cell_counts_of_the_recorded_configuration():
    regions = build_regions(11, T, 3, 5)
    counts = [len(regions[r].cells) for r in range(2, 12)]
    assert counts == [3, 18, 81, 324, 351, 378, 405, 432, 459, 486]  # the table of Section 7.6
    assert len(regions[11].facets) == 4347 and len(regions[11].points) == 92


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_kuhn_subdivision_restricts_to_every_facet(d, n):
    """The claim after eq:subdivision: the restriction to a facet is the subdivision of the facet with inherited order."""
    simplex = tuple(tuple(Fraction(int(i == j)) for i in range(d + 1)) for j in range(d + 1))  # standard simplex, vertex order e_0, ..., e_d
    cells = kuhn_subdivision(simplex, n)
    assert len(cells) == n**d == len(kuhn_paths(n, d))

    for k in range(d + 1):
        facet = simplex[:k] + simplex[k + 1 :]
        on_facet = set()
        for cell in cells:
            points = [point for point in cell if point[k] == 0]
            if len(points) == d:
                on_facet.add(frozenset(tuple(x for i, x in enumerate(point) if i != k) for point in points))
        assert (
            on_facet == set(kuhn_subdivision(tuple(tuple(x for i, x in enumerate(p) if i != k) for p in facet), n)) if d > 1 else len(on_facet) == 1
        )


def test_boundary_facets_and_child_matching():
    regions = build_regions(6, T, 2, 4)
    for r in range(3, 7):
        g, child = regions[r], regions[r - 1]
        coordinate = [f for f, kind in g.boundary.items() if kind == COORDINATE]
        assert len(coordinate) == len(child.cells)
        # On O_r the terminal facet is an (r - 2)-simplex cut into n^(r - 2) cells for r <= r_0, and into n^(r_0 - 2) afterwards.
        assert sum(kind == TERMINAL for kind in g.boundary.values()) == 2 ** min(r, 4) // 4
    # O_2 is the interval [u_2, v_2]: its two endpoints are its equality facet and its terminal facet.
    assert sorted(map(str, regions[2].boundary.values())) == ["0", TERMINAL]
