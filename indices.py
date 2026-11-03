"""Index conventions shared by every module (Sections 7.2, 7.4 and 7.5, and the table format in README.md).

A quadratic on a simplex with m ordered vertices is stored in the Bernstein basis
eq:quadraticbasis by its coefficients at the pairs st, s <= t, of vertex positions.
The pairs are ordered: the repeated pairs ss first, then the distinct pairs s < t
lexicographically.  Coordinate pairs ij, i < j, index the directions e_i - e_j of
the weights and are ordered lexicographically as well.
"""

import itertools

import numpy as np


def coefficient_pairs(m):
    """Pairs st, s <= t, of positions 0, ..., m - 1 in storage order: ss first, then s < t lexicographic.

    A cell of O_r has m = r vertices and r (r + 1) / 2 coefficients, a facet has m = r - 1
    vertices and r (r - 1) / 2 coefficients.
    """
    return [(s, s) for s in range(m)] + list(itertools.combinations(range(m), 2))


def coordinate_pairs(r):
    """Coordinate pairs ij, i < j, of {0, ..., r - 1} in storage order, one per direction e_i - e_j."""
    return list(itertools.combinations(range(r), 2))


def subsets_containing_zero(r):
    """The nonempty proper subsets I of {0, ..., r - 1} with 0 in I, as 0/1 rows of a (2^(r - 1) - 1) x r array.

    Capacity for I and for its complement is the same inequality (Section 7.5), so only
    these subsets are checked.
    """
    masks = []
    for size in range(1, r):
        for subset in itertools.combinations(range(1, r), size - 1):
            mask = np.zeros(r, dtype=np.int64)
            mask[0] = 1
            mask[list(subset)] = 1
            masks.append(mask)
    return np.array(masks, dtype=np.int64).reshape(-1, r)


def crossing_matrix(r):
    """0/1 array with entry [I, ij] = 1 when the direction e_i - e_j crosses the subset I.

    Row I of subsets_containing_zero(r), column ij of coordinate_pairs(r).  The weight crossing
    I at a coefficient, sum of w_ij over i in I, j not in I, is this matrix applied to W.
    """
    masks = subsets_containing_zero(r)
    pairs = coordinate_pairs(r)
    return np.array([[int(mask[i] != mask[j]) for i, j in pairs] for mask in masks], dtype=np.int64).reshape(len(masks), len(pairs))
