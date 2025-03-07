"""Fitting functions themselves an some optimization helpers."""

from ..common import np, sp, sc
from . import ModelLite, Metrics, FittingOptions, Spectrum, PolySpectrum


def poly_fit_(data: np.ndarray, options: FittingOptions) -> ModelLite:
    """Fits this model using LMS method. Uses some infrastructure.

    Arguments:
        data (np.ndarray): training dataset, it is required for the fitting to work.
        options (FittingOptions): An object that contains necessary configuration:
            rank (int): manually specifies polynomial rank for fitting. Attempts to
            figure out the required rank if value is set to default of "0".
            raise_rank (boolean): enables or disables automatic rank raise algorithm. By
            default is set to `False`.

    Returns:
        ModelLite: Vector of the fitting results.
    """
    data_x: np.ndarray = np.arange(data.size)
    rank = options.rank
    if options.raise_rank:
        rank = find_poly_rank(data)
        print(f"Model rank was raised to {rank}")
    poly_c = np.polyfit(data_x, data, rank)
    poly_f = np.poly1d(poly_c)
    poly_s = PolySpectrum(rank, str(options.var_main))
    return ModelLite(
        str(poly_s.var_main), poly_s.expr_raw, poly_s.expr_sp, poly_f, poly_c
    )


def poly_fit_lite(data: np.ndarray, rank: int) -> np.ndarray:
    """Lightweight version of `poly_fit` without library infrastructure.
    Pure numpy and stuff.

    Arguments:
        data (np.ndarray): training dataset, it is required for the fitting to work.
        rank (int): specifies the ranked polynomial to fit.

    Returns:
        np.ndarray: Vector of the fitted results.
    """
    data_x = np.arange(data.size)
    poly_c = np.polyfit(data_x, data, rank)
    poly_f = np.poly1d(poly_c)
    return poly_f(data_x)


def nonline_fit_(data: np.ndarray, options: FittingOptions) -> ModelLite:
    """Internal call to `nonline_fit_` that fits this model using experimental
    methods from this library. It can update internal fields of the instance,
    after which it qualifies as being fitted.

    Arguments:
        data (np.ndarray): training dataset, it is required for the fitting to work.
        options (FittingOptions): An instance that describes how to conduct the
        operation. Uses default options if not specified.
    Returns:
        ModelLite: Object for the achieved fit.
    """
    nonline = Spectrum(options.expr_raw, str(options.var_main))
    rank = options.rank
    if not options.raise_rank:
        rank = find_poly_rank(data, rank)
        if rank != nonline.expr_rank:
            if any(d == 0 for d in nonline.ranked(rank)):
                rank = nonline.expr_rank
                print(
                    f"Failed model rank raise, proceeding with original value of {rank}"
                )
            else:
                print(f"Model rank was raised to {rank}")

    poly = PolySpectrum(rank, str(options.var_main))
    poly_s = poly.ranked(rank)

    data_x = np.arange(data.size)
    poly_c = np.polyfit(data_x, data, rank - 1)[::-1]

    balance: list[sp.Expr] = []
    nonline_s = nonline.ranked(rank)
    if any(d == 0 for d in nonline_s):
        raise RuntimeError("Cannot solve underdetermined system.")
    for i, a in enumerate(nonline):
        balance.append((a - poly_s[i]).subs(poly.expr_coeffs[i], poly_c[i]))
    if rank == len(nonline.expr_coeffs):
        solution = sp.nonlinsolve(balance, nonline.expr_coeffs).args[0]
    else:
        solution0 = sp.nonlinsolve(
            balance[: len(nonline.expr_coeffs)], nonline.expr_coeffs
        ).args[0]
        solution = nonline_lsm(balance, nonline.expr_coeffs, solution0)

    if options.numeric_optimize:
        solution = numeric_optimize(data, solution, nonline)
    return nonline.apply_trained(solution)


def find_poly_rank(data: np.ndarray, rank: int = 0, rank_range: int = 12) -> int:
    """Uses experimental algorithm and criteria to figure out the required
    polynomial rank to adequately fit the given data. Can be costly so it is
    togglable in `FittingOptions`.
    This method is very susceptible to internal noises and errors of the data.

    Arguments:
        data (np.ndarray): given values to analyse.
        rank (int): base value for the rank, used as a starting point. Defaults to 3
        if nothing is specified.
        rank_range (int): ceiling for the rank search. Defaults to 12, may be overridden
        for big datasets.

    Returns:
        int: Numeric value which represents possibly optimal rank.
    """
    rank = rank if rank != 0 else 3
    rank_rse = np.zeros(rank_range)
    rank_rsq = np.zeros(rank_range)

    prev_corr = 0
    prev_diff = 0
    for i in range(rank_range):
        data_y = poly_fit_lite(data, rank + i)

        corr = np.corrcoef(data, data_y)[0, 1]
        diff = np.abs(corr - prev_corr)
        if prev_diff < diff:
            return rank + i
        prev_corr = corr

        rank_rse[i] = Metrics.rse(data, data_y)
        rank_rsq[i] = Metrics.r_sq(data, data_y)

    rank_rr = []
    for i in range(rank_range - 1):
        rank_rr.append(
            np.abs(rank_rse[i + 1] - rank_rse[i])
            * np.abs(rank_rsq[i + 1] - rank_rsq[i])
        )
    for i, r in enumerate(rank_rr):
        if r < 10**-6:
            return rank + i

    return rank


def nonline_lsm(
    system: list[sp.Expr], coeffs: list[sp.Symbol], solution0: np.ndarray
) -> np.ndarray:
    """Nonlinear Least Squares Method that can be used to solve overdefined systems
    of equations. Utilizes numeric algorithms and thus requires starting values.

    Arguments:
        system (list[sp.Expr]): list of symbolic expressions that represent the system of equations.
        The system may be bigger than given coeffs.
        coeffs (list[sp.Symbol]): list of the symbol coefficients to solve around.
        solution0 (np.ndarray): an `ndarray` of starting values for the coefficients.

    Returns:
        np.ndarray: array of the solution that was optimized using possibly overdefined
        system of equations.
    """
    solution0 = np.array(solution0).astype(np.float64)

    for i, eq in enumerate(system):
        system[i] = eq.subs(sp.abc.H, 1)
    func = sp.lambdify(coeffs, system, "scipy")

    def wrapper(args: np.ndarray) -> np.float64:
        return func(*args)

    solution = sc.optimize.least_squares(wrapper, solution0, method="lm")
    return np.array(solution.x).astype(np.float64)


def numeric_optimize(
    data: np.ndarray, solution: np.ndarray, spectrum: Spectrum
) -> np.ndarray:
    """Utilizes numeric methods to optimize given solution using original data.
    Requires a spectrum object and already calculated solution to operate.

    Arguments:
        data (np.ndarray): original data that was used for fitting the solution.
        solution (np.ndarray): vector of calculated values that need improving.
        spectrum (Spectrum): an object that represents model through differential
        discretes.

    Returns:
        np.ndarray: New vector with optimized solution.
    """
    data_x = np.arange(data.size)
    factors = spectrum.expr_coeffs.copy()
    factors.insert(0, spectrum.var_main)
    solution = sc.optimize.curve_fit(
        spectrum.apply_trained(solution).model,
        data_x,
        data,
        p0=solution,
        maxfev=100000,
    )[0]
    return solution
