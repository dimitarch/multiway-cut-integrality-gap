"""The integer table of a flow instance (Section 7.4) and its file format (README.md).

A table holds, for every r = 2, ..., q, the subdivision of O_r and three integer arrays:

    H[i, f, k]   H_{i, f, st}: label i, facet f, coefficient k of the pair st of vertices of f
    Z[P, i, e]   Z_{i, P, st}: cell P, label i, pair e = st of cell vertices with s < t
    W[P, k, e]   W_{P, st, ij}: cell P, coefficient k of the pair st of cell vertices, direction e = ij

together with the denominator D and the threshold T.  The file is a NumPy .npz archive of
integer arrays: the scalars q, n, r_0, D, T_num, T_den and, for every r, Q_r, points_r,
cells_r, facets_r, H_r, Z_r, W_r.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from geometry import Region, build_regions
from indices import coefficient_pairs, coordinate_pairs


@dataclass
class RegionTable:
    """The data of a flow instance on one ordered region O_r, unscaled (Section 7.1)."""

    region: Region
    H: np.ndarray
    Z: np.ndarray
    W: np.ndarray

    def __post_init__(self):
        g = self.region
        r = g.r
        self.H, self.Z, self.W = (np.asarray(a, dtype=np.int64) for a in (self.H, self.Z, self.W))
        assert self.H.shape == (r, len(g.facets), len(coefficient_pairs(g.d))), ("H", self.H.shape)
        assert self.Z.shape == (len(g.cells), r, len(coordinate_pairs(r))), ("Z", self.Z.shape)
        assert self.W.shape == (len(g.cells), len(coefficient_pairs(r)), len(coordinate_pairs(r))), ("W", self.W.shape)
        assert self.W.min(initial=0) >= 0, "weights must be nonnegative"


@dataclass
class FlowTable:
    q: int
    n: int
    r_0: int
    D: int
    T: Fraction
    regions: dict  # r -> RegionTable

    def save(self, path):
        arrays = dict(q=self.q, n=self.n, r_0=self.r_0, D=self.D, T_num=self.T.numerator, T_den=self.T.denominator)
        for r, rt in self.regions.items():
            g = rt.region
            arrays[f"Q_{r}"] = g.Q
            arrays[f"points_{r}"] = g.p
            arrays[f"cells_{r}"] = np.array([cell.vertices for cell in g.cells], dtype=np.int64)
            arrays[f"facets_{r}"] = np.array(g.facets, dtype=np.int64)
            arrays[f"H_{r}"], arrays[f"Z_{r}"], arrays[f"W_{r}"] = rt.H, rt.Z, rt.W
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        """Read a table and rebuild its geometry from the stored points, cells and facets.

        The stored subdivision of every O_r is compared with the construction of Section 7.2,
        so that the only data a table contributes are its indexing and the integers H, Z, W.
        """
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for old, new in LEGACY_NAMES.items():
            for key in list(data):
                if key == old or key.startswith(old + "_"):
                    data[new + key[len(old) :]] = data.pop(key)
        assert all(a.dtype.kind in "iu" for a in data.values()), "a table holds integers only"

        scalar = lambda key: int(data[key])
        q, n, r_0, D = scalar("q"), scalar("n"), scalar("r_0"), scalar("D")
        T = Fraction(scalar("T_num"), scalar("T_den"))
        assert q >= 2 and n >= 1 and 2 <= r_0 <= q and D >= 1 and Fraction(1, 2) < T < 1

        canonical = build_regions(q, T, n, r_0)
        regions = {}
        for r in range(2, q + 1):
            Q = scalar(f"Q_{r}")
            points = [tuple(Fraction(int(x), Q) for x in row) for row in data[f"points_{r}"]]
            region = Region(r, T, points, [tuple(map(int, row)) for row in data[f"cells_{r}"]], [tuple(map(int, row)) for row in data[f"facets_{r}"]])
            assert region.Q == Q, f"Q_{r} is not the least common denominator of the points"
            assert region.cell_point_sets() == canonical[r].cell_point_sets(), f"the stored cells of O_{r} are not the subdivision of Section 7.2"
            if r >= 3:
                region.check_child(regions[r - 1].region)
            regions[r] = RegionTable(region, data[f"H_{r}"], data[f"Z_{r}"], data[f"W_{r}"])

        return cls(q, n, r_0, D, T, regions)


# Names used by the archives produced before this code; accepted on load so that they can be replayed.
LEGACY_NAMES = {"k": "q", "base": "r_0", "faces": "facets", "flux": "H", "edge": "Z", "weight": "W"}
