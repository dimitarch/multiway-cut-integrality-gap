"""A sparse linear program and its solution by HiGHS or Gurobi.

    minimize c . x   subject to   row_lower <= A x <= row_upper,   lower <= x <= upper.

Both solvers receive exactly the same columns, rows, bounds and objective.  Every row is
divided by its largest absolute coefficient, which keeps the matrix well scaled for the
interior point methods without changing the feasible set.
"""

from array import array
import math

import numpy as np
from scipy.sparse import coo_matrix

INFINITY = math.inf


class LinearProgram:
    """Columns are created by `variable`, rows by `at_most` and `equal`.

    A linear expression is a dict {column: coefficient}.  Expressions are combined with `add`.
    """

    def __init__(self):
        self.lower, self.upper = array("d"), array("d")
        self.row_lower, self.row_upper = array("d"), array("d")
        self.rows, self.columns, self.values = array("q"), array("q"), array("d")
        self.objective = {}

    @property
    def num_columns(self):
        return len(self.lower)

    @property
    def num_rows(self):
        return len(self.row_lower)

    @property
    def num_nonzeros(self):
        return len(self.values)

    def variable(self, lower=-INFINITY, upper=INFINITY):
        self.lower.append(lower)
        self.upper.append(upper)
        return self.num_columns - 1

    def variables(self, shape, lower=-INFINITY, upper=INFINITY):
        """An integer array of the given shape holding fresh column indices."""
        size = int(np.prod(shape))
        first = self.num_columns
        self.lower.extend([lower] * size)
        self.upper.extend([upper] * size)
        return np.arange(first, first + size, dtype=np.int64).reshape(shape)

    def fix(self, column, value):
        self.lower[column] = self.upper[column] = float(value)

    def add_row(self, expression, lower, upper):
        expression = {int(j): float(a) for j, a in expression.items() if abs(a) > 1e-15}  # as the earlier implementation: drop floating-point noise
        finite_bounds = [abs(b) for b in (lower, upper) if math.isfinite(b)]
        if not expression:
            assert all(b == 0 for b in finite_bounds), "an empty row with a nonzero bound is infeasible"
            return

        scale = max(max(abs(a) for a in expression.values()), *finite_bounds, 0.0) or 1.0
        i = self.num_rows
        self.row_lower.append(lower / scale)
        self.row_upper.append(upper / scale)
        for j, a in expression.items():
            self.rows.append(i)
            self.columns.append(j)
            self.values.append(a / scale)

    def at_most(self, expression, bound=0.0):
        self.add_row(expression, -INFINITY, bound)

    def equal(self, expression, value=0.0):
        self.add_row(expression, value, value)

    def cost_vector(self):
        c = np.zeros(self.num_columns)
        for j, a in self.objective.items():
            c[j] += a
        return c

    def matrix(self):
        return coo_matrix(
            (np.frombuffer(self.values), (np.frombuffer(self.rows, dtype=np.int64), np.frombuffer(self.columns, dtype=np.int64))),
            shape=(self.num_rows, self.num_columns),
        ).tocsc()

    def violation(self, x):
        """Largest violation of a scaled row or of a column bound by the vector x."""
        activity = self.matrix() @ x
        rows = np.maximum(np.frombuffer(self.row_lower) - activity, activity - np.frombuffer(self.row_upper))
        columns = np.maximum(np.frombuffer(self.lower) - x, x - np.frombuffer(self.upper))
        return float(max(rows.max(initial=0.0), columns.max(initial=0.0)))


def add(expression, other, factor=1.0):
    """expression += factor * other, in place; returns expression."""
    for j, a in other.items():
        expression[j] = expression.get(j, 0.0) + factor * a
    return expression


# ---- HiGHS ----

HIGHS_METHODS = ("ipm", "hipo", "simplex", "pdlp")


def solve_with_highs(
    lp, method="ipm", threads=0, time_limit=INFINITY, feasibility=1e-9, optimality=1e-7, crossover=False, log_file=None, options=None
):
    """Solve with HiGHS.  Returns (x or None, information dict).

    method       ipm    the serial interior point solver IPX
                 hipo   the parallel interior point solver, available when highspy-extras is installed
                 simplex, pdlp
    feasibility  primal feasibility tolerance; the rounding step repairs small violations, but a
                 tighter tolerance means fewer repairs and a better value
    optimality   dual feasibility and relative optimality tolerance; the value is off by about this much
    options      extra HiGHS options, name -> value, applied last
    """
    import highspy

    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = lp.num_columns, lp.num_rows
    model.col_cost_ = lp.cost_vector()
    model.col_lower_, model.col_upper_ = np.frombuffer(lp.lower), np.frombuffer(lp.upper)
    model.row_lower_, model.row_upper_ = np.frombuffer(lp.row_lower), np.frombuffer(lp.row_upper)
    A = lp.matrix()
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.num_col_, model.a_matrix_.num_row_ = lp.num_columns, lp.num_rows
    model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = A.indptr, A.indices, A.data

    highs = highspy.Highs()
    settings = dict(
        solver=method,
        run_crossover="on" if crossover else "off",
        presolve="on",
        parallel="on",
        threads=threads,
        time_limit=float(time_limit),
        primal_feasibility_tolerance=feasibility,
        dual_feasibility_tolerance=optimality,
        ipm_optimality_tolerance=optimality,
    )
    if log_file:
        settings["log_file"] = str(log_file)
    settings.update(options or {})
    for name, value in settings.items():
        if highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
            hint = "; install highspy-extras for hipo" if name == "solver" and value == "hipo" else ""
            raise SystemExit(f"HiGHS rejected the option {name} = {value!r}{hint}")
    highs.passModel(model)
    highs.run()

    info = highs.getInfo()
    solution = highs.getSolution()
    result = dict(
        status=highs.modelStatusToString(highs.getModelStatus()),
        objective=info.objective_function_value,
        ipm_iterations=info.ipm_iteration_count,
        simplex_iterations=info.simplex_iteration_count,
        pdlp_iterations=info.pdlp_iteration_count,
        max_primal_infeasibility=info.max_primal_infeasibility,
        max_dual_infeasibility=info.max_dual_infeasibility,
    )
    return (np.array(solution.col_value) if solution.value_valid else None), result


# ---- Gurobi ----

GUROBI_METHODS = ("barrier", "pdhg", "pdhg-gpu", "concurrent", "simplex")


def solve_with_gurobi(
    lp, method="barrier", threads=0, time_limit=INFINITY, feasibility=1e-9, optimality=1e-7, crossover=False, log_file=None, options=None
):
    """Solve with Gurobi (gurobipy with a license; PDHG needs Gurobi 12, PDHG on a GPU needs Gurobi 13 on Linux).

    method       barrier (Method=2, the counterpart of the HiGHS interior point solvers), pdhg (Method=6),
                 pdhg-gpu (Method=6 with PDHGGPU=1), concurrent (Method=3), simplex (Method=1)
    feasibility  FeasibilityTol, or the PDHG tolerances
    optimality   BarConvTol and OptimalityTol, or PDHGConvTol
    options      extra Gurobi parameters, name -> value, applied last
    """
    import gurobipy as gp
    from gurobipy import GRB

    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 1)
    if log_file:
        environment.setParam("LogFile", str(log_file))
    environment.start()

    model = gp.Model(env=environment)
    finite = lambda bounds: np.clip(np.frombuffer(bounds), -GRB.INFINITY, GRB.INFINITY)
    x = model.addMVar(lp.num_columns, lb=finite(lp.lower), ub=finite(lp.upper), obj=lp.cost_vector())
    row_lower, row_upper = np.frombuffer(lp.row_lower), np.frombuffer(lp.row_upper)
    equal = row_lower == row_upper
    assert np.all(equal | np.isneginf(row_lower)), "rows are either equalities or upper bounds"
    model.addMConstr(lp.matrix().tocsr(), x, np.where(equal, GRB.EQUAL, GRB.LESS_EQUAL), row_upper)
    model.ModelSense = GRB.MINIMIZE

    gpu = method == "pdhg-gpu"
    settings = dict(
        Method=dict(barrier=2, pdhg=6, concurrent=3, simplex=1)["pdhg" if gpu else method], Crossover=-1 if crossover else 0, Threads=threads
    )
    if math.isfinite(time_limit):
        settings["TimeLimit"] = float(time_limit)
    if method.startswith("pdhg"):
        settings.update(PDHGGPU=int(gpu), PDHGAbsTol=max(feasibility, 1e-9), PDHGRelTol=feasibility, PDHGConvTol=optimality)
    else:
        settings.update(FeasibilityTol=max(feasibility, 1e-9), OptimalityTol=max(optimality, 1e-9), BarConvTol=optimality)
    settings.update(options or {})
    for name, value in settings.items():
        try:
            model.setParam(name, value)
        except gp.GurobiError as error:
            raise SystemExit(f"Gurobi rejected the parameter {name} = {value!r}: {error}") from error
    model.optimize()

    status_names = {getattr(GRB.Status, name): name for name in dir(GRB.Status) if name.isupper()}
    result = dict(
        status=status_names.get(model.Status, str(model.Status)),
        runtime=model.Runtime,
        barrier_iterations=model.BarIterCount,
        simplex_iterations=int(model.IterCount),
    )
    solution = None
    if model.SolCount > 0:
        solution = np.array(x.X)
        result["objective"] = model.ObjVal
        for key, attribute in (("max_constraint_violation", "ConstrVio"), ("max_bound_violation", "BoundVio"), ("max_dual_violation", "DualVio")):
            try:
                result[key] = getattr(model, attribute)
            except (AttributeError, gp.GurobiError):
                pass
    model.dispose()
    environment.dispose()
    return solution, result


def parse_options(pairs):
    """['hipo_block_size=256', 'BarOrder=1', 'run_crossover=on'] -> {name: typed value}."""
    options = {}
    for pair in pairs or []:
        name, _, text = pair.partition("=")
        value = text
        for cast in (int, float):
            try:
                value = cast(text)
                break
            except ValueError:
                continue
        if text.lower() in ("true", "false"):
            value = text.lower() == "true"
        options[name.strip()] = value
    return options
