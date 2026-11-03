# Multiway Cut gap 1.2325: search, rounding and exact verification

Code accompanying the paper *An Integrality-Gap Lower Bound of 1.2325 for Minimum Multiway Cut*.
The lower bound `1.2325` on the integrality gap of the CKR relaxation rests on one integer table,
`data/q11_table_trim.npz`, a flow instance on the 11-vertex simplex.  This repository contains the
program that checks that table exactly and the programs that found it.

* `verify.py` reads a table, checks every condition the proof needs in integer arithmetic, and
  prints the exact fractions `B`, `R`, `L` and the value `(B - R) / L`.  This is the only part the
  proof depends on.  It imports no optimization software.
* `search.py` builds the linear program over the family of flow instances and solves it in
  floating point with HiGHS or Gurobi.
* `rounding.py` turns the floating solution into an integer table and lowers its weights.
* `affine.py` writes the affine flow instances as tables; they verify to closed forms and serve as
  a test of the verifier.

## Installation

This is a [uv](https://docs.astral.sh/uv/) project.  Python 3.10 or newer; uv downloads one if
none is installed.  In the repository root:

```bash
uv sync
```

installs numpy, scipy and highspy, which is everything `verify.py`, `rounding.py`, `affine.py` and
the HiGHS search need.  Optional extras:

```bash
uv sync --extra hipo     # highspy-extras: the parallel interior point solver HiPO of HiGHS
uv sync --extra gurobi   # gurobipy: the Gurobi backend (needs a license, see below)
uv sync --all-extras     # both
```

pytest and ruff are in the `dev` group, which `uv sync` installs by default.  Without uv,
`pip install numpy scipy highspy` suffices for everything except the optional solvers.

Every command below is run from the repository root with `uv run`, and every script prints its
usage with `-h`.

## Verify the table

```bash
uv run python verify.py data/q11_table_trim.npz --report report.json
```

takes about 10 seconds and prints, among per-region progress lines,

```
"passed": true,
"cells": 2937,
"B": "1725000000167/1725000000000",
"R": "2593033/910800000000000",
"L": "43107286191734959/53130000000000000",
"decimal": 1.232506254686709,
"capacity_checks": 53496999,
"sha256": "8c311470822b74acbde2e2af0fb1b96cdc4db44b192662c1ba3c51ccbafa71f8"
```

The verifier rebuilds the subdivision from the parameters `q`, `n`, `r_0`, `T` and compares it with
the vertices and cells stored in the table, checks the two boundary rules and the integer capacity
inequality at every cell, quadratic coefficient and subset, computes the allowances as the
conservation residuals, and sums `B`, `R`, `L` as exact fractions.  A failing check names the
region, cell, coefficient and subset, and the exit code is nonzero.  The bulk integer work runs in
64-bit NumPy arrays after an explicit bound excludes overflow; all sums use Python integers and
fractions.

The affine instances give a second check with known answers:

```bash
uv run python affine.py 7 3/5 1 2 affine.npz      # q = 7, T = 3/5, n = 1, r_0 = 2
uv run python verify.py affine.npz                # B, R, L equal the closed forms; value 1.2208...
```

## Regenerate the table

### From the stored floating solution

`data/q11_gurobi.npz` is the solution of the linear program from which the table was rounded.
Rounding and trimming are deterministic, so

```bash
uv run python rounding.py round data/q11_gurobi.npz data/q11_table.npz --D 230000000000
uv run python rounding.py trim data/q11_table.npz data/q11_table_trim.npz
```

reproduces both stored tables byte for byte in about a minute.  `round` rounds
the coefficients to `integers / D`, imposes the boundary rules exactly, rounds the weights upward
and raises them until every capacity inequality holds.  `trim` lowers the weights direction by
direction as long as every inequality stays valid, which lowers `L` and leaves the flows
unchanged.  Both run the verifier at the end and write its report to `<table>.npz.json`.  A larger
`D` loses less in rounding; the verifier refuses a `D` for which the 64-bit bound fails.

### From scratch

```bash
uv run python search.py 11 3 249/400 --r_0 5 --common-quadratic 6-10 --common-affine 11 \
    --backend gurobi --threads 8 --out data/q11_gurobi
```

The positional arguments are `q`, the subdivision parameter `n` and the threshold `T`.  Regions
`O_r` with `r <= r_0` are subdivided into `n^(r - 1)` cells per initial simplex, higher ones by
cones.  Each region has one of three types:

* `free` (default): every label has its own quadratic flow, the weights are free;
* `common-quadratic`: all labels share one quadratic flow plus constant offsets `A (e_i - u_r)`,
  every weight is `A / (r - 1)`;
* `common-affine`: the same with an affine shared flow.

The program has the allowance set to zero, normalizes `B = 1` and minimizes `L`, so the floating
value is `1 / L`; `--allowance` makes the allowances unknowns with normalization `B - R = 1`.
The run above has 712,460 columns, 3.9 million rows and 22 million nonzeros.  It builds in about
20 seconds and Gurobi's barrier with 8 threads solves it in about 19 minutes on a Mac
(value 1.2325062 before rounding).  `--build-only` prints the size and stops.  The output is
`STEM.npz` (the floating solution), `STEM.json` (sizes, solver status, violations, `L` and the
value) and `STEM.log` (the solver log).

Regions can be held at the values of an earlier solution with the same parameters and types, to
reoptimize the others:

```bash
uv run python search.py 11 3 249/400 --r_0 5 --common-quadratic 6-10 --common-affine 11 \
    --fix-from data/q11_gurobi.npz --fix 2-5,10-11 --out data/q11_b
```

### Solver options

Both backends receive the identical program (`lp.py`), with every row divided by its largest
coefficient, and the result is rounded and verified the same way, so the solver never enters the
proof.

| option | HiGHS (`--backend highs`, default) | Gurobi (`--backend gurobi`) |
|---|---|---|
| `--method` | `ipm` (default, serial IPX), `hipo` (parallel, needs `--extra hipo`), `simplex`, `pdlp` | `barrier` (default), `pdhg`, `pdhg-gpu`, `concurrent`, `simplex` |
| `--threads N` | `0` lets the solver choose | same |
| `--time-limit S` | seconds; an interior point run stopped by the limit returns no solution | seconds |
| `--feasibility` (default `1e-9`) | `primal_feasibility_tolerance` | `FeasibilityTol`, or `PDHGAbsTol`, `PDHGRelTol` |
| `--optimality` (default `1e-7`) | `dual_feasibility_tolerance`, `ipm_optimality_tolerance` | `BarConvTol`, `OptimalityTol`, or `PDHGConvTol` |
| `--crossover` | crossover after the interior point method (slow, not needed) | `Crossover=-1` (default `0`) |
| `--option NAME=VALUE` | any HiGHS option, repeatable | any Gurobi parameter, repeatable |

A tighter feasibility tolerance means fewer repairs in rounding and a better table; the value is
off by about the optimality tolerance.  Gurobi's PDHG defaults are `1e-6`, so pass
`--feasibility 1e-6 --optimality 1e-6` for PDHG screening runs.  `pdhg-gpu` needs Gurobi 13 on
Linux with an NVIDIA GPU; check in the log that PDHG actually ran on the GPU.

Gurobi license: the license bundled with `gurobipy` is limited to 2000 rows and columns, so the
full program needs a full or Web License Service license.  Gurobi reads `gurobi.lic` from the home
directory or from the path in `GRB_LICENSE_FILE`; `grbgetkey <key>` writes it.  Without a valid
license the script stops at `gp.Env`.

## Files

| file | content |
|---|---|
| `verify.py` | the exact checks and `B`, `R`, `L` |
| `table.py` | the integer table `(H, Z, W)` and its `.npz` format |
| `reconstruction.py` | the integer maps from `(H, Z)` to flow coefficients and divergences, and the exact inverse |
| `geometry.py` | the ordered regions `O_r`, their subdivision into cells, facets, orientations, volumes, boundary facets |
| `indices.py` | the order of the quadratic coefficients `st`, the directions `ij` and the subsets `I` |
| `search.py` | the linear program and its command line |
| `lp.py` | a sparse linear program solved by HiGHS or Gurobi |
| `rounding.py` | rounding to an integer table, weight repair, weight trimming |
| `affine.py` | the affine flow instances as tables |
| `tests/` | `uv run pytest` |
| `data/q11_table_trim.npz` | the table; `.npz.json` is its verification report |
| `data/q11_table.npz` | the table before trimming; `.npz.json` is its verification report |
| `data/q11_gurobi.npz`, `.json`, `.log` | the floating solution of the linear program, its summary and the solver log |

## Table format

A table is a NumPy `.npz` archive of integer arrays: the scalars `q`, `n`, `r_0`, `D`, `T_num`,
`T_den` and, for every `r = 2, ..., q`,

| array | shape | content |
|---|---|---|
| `Q_r` | | common denominator of the vertex coordinates of `O_r` |
| `points_r` | vertices x r | integer coordinate numerators `p_{s,j}`, one row per vertex |
| `cells_r` | cells x r | increasing vertex indices of every cell |
| `facets_r` | facets x (r - 1) | increasing vertex indices of every facet; a facet is oriented outward from its first incident cell |
| `H_r` | r x facets x r(r-1)/2 | `H_{i,f,st}` |
| `Z_r` | cells x r x r(r-1)/2 | `Z_{i,P,st}`, `s < t` |
| `W_r` | cells x r(r+1)/2 x r(r-1)/2 | `W_{P,st,ij}`, nonnegative |

The coefficient pairs `st` are ordered with the repeated pairs `ss` first and then the pairs
`s < t` lexicographically; the directions `ij`, `i < j`, lexicographically (`indices.py`).

## Tests and formatting

```bash
uv run pytest
```

checks the geometry (cell counts, facet restriction, boundary facets), the verifier against the
closed forms of the affine instances for several `(q, T, n, r_0)`, and the search, rounding and
verification pipeline on small configurations with HiGHS.

```bash
uv run ruff format . && uv run ruff check .
```

formats and lints with the configuration in `pyproject.toml`.
