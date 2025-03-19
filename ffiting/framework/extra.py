"""Fitting functions themselves an some optimization helpers."""

from sympy.abc import H

from ffiting.framework.options import FittingModes
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
        rank = find_poly_rank(data, rank)
        print(f"Model rank was raised to {rank}")
    poly_c = np.polyfit(data_x, data, rank)
    poly_f = np.poly1d(poly_c)
    poly_s = PolySpectrum(rank, str(options.var_main))
    return ModelLite(
        str(poly_s.var_main_sp), poly_s.expr_raw, poly_s.expr_sp, poly_f, poly_c
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
    nonline = Spectrum(options.expr_raw, options.var_main)
    if any(d == 0 for d in nonline.ranked()):
        raise RuntimeError("Cannot solve underdetermined system.")

    rank = options.rank if options.rank >= nonline.expr_rank else nonline.expr_rank
    if options.raise_rank:
        rank_new = find_poly_rank(data, rank)
        if rank_new != nonline.expr_rank:
            rank = rank_new
            print(f"Polynomial model rank was raised to {rank}.")
        else:
            print(f"Polynomial model rank was not raised, value is {rank}.")
    poly = PolySpectrum(rank, options.var_main)

    data_x = np.arange(data.size)
    poly_c = np.polyfit(data_x, data, rank - 1)[::-1]

    if options.fitting_mode == FittingModes.DSB:
        balance = dsb_(poly, nonline)

        print("The balance was formed:")
        for expr in balance:
            display(expr)

        if rank == nonline.expr_rank:
            solution = sp.nonlinsolve(balance, nonline.expr_coeffs).args[0]

            for i in np.arange(poly.expr_rank):
                solution = solution.subs(poly.expr_coeffs[i], poly_c[i])
            
            print("Solutions were found:")
            display(solution)
            
        else:
            for i in np.arange(poly.expr_rank):
                for j in np.arange(poly.expr_rank):
                    balance[i] = balance[i].subs(poly.expr_coeffs[j], poly_c[j])
            
            solution0 = sp.nonlinsolve(
                balance[: nonline.expr_rank], nonline.expr_coeffs
            )

            print("Solutions were found:")
            for sol in solution0.args:
                display(sol)

            solution0 = solution0.args[0]
            
            solution = nonline_lsm(balance, nonline.expr_coeffs, solution0)

            print("Solution was rank optimized:")
            display(solution)

    elif options.fitting_mode == FittingModes.DSBI:
        balance = dsb_i_(poly, nonline, poly_c, 1.002)

        print("The balance was formed:")
        for expr in balance:
            display(expr)

        solution = sp.nonlinsolve(balance, nonline.expr_coeffs)

        print("Solutions were found:")
        for expr in solution.args:
            display(expr)

        solution = solution.args[-1]
    else:
        raise RuntimeError("Unrecognized fitting mode was passed.")

    if options.numeric_optimize:
        solution = numeric_optimize(data, solution, nonline)

        print("Solution was numerically optimized:")
        display(solution)

    return nonline.apply_trained(solution)


def dsb_(poly: Spectrum, nonline: Spectrum) -> list[sp.Expr]:
    """Implementation of differential spectra balance creation using "strong"
    criteria for difference minimization. It can be applied with for most basic
    datasets and models with good enough results.

    Args:
        poly (Spectrum): Spectrum of the polynomial model.
        nonline (Spectrum): Spectrum of the nonline model.

    Returns:
        list[sp.Expr]: A differential spectra balance that was created.
    """
    nonline_s = nonline.ranked()
    poly_s = poly.ranked()

    def ns(i: int) -> sp.Expr:
        return nonline_s[i] if i < nonline.expr_rank else 0

    balance: list[sp.Expr] = []
    for i, c in enumerate(poly_s):
        balance.append(c - ns(i))

    return balance


def dsb_i_(
    poly: Spectrum, nonline: Spectrum, poly_c: np.ndarray, h: float
) -> list[sp.Expr]:
    """Method for creation of the differential spectra balance using the "soft"
    difference minimization criteria. It theoretically can be applied for more
    broad array of data and models due to being more lenient.

    Args:
        poly (Spectrum): Spectrum of the polynomial model.
        nonline (Spectrum): Spectrum of the nonline model.
        poly_c (np.ndarray): Container with calculated polynomial coefficients.
        h (float): Calculated value for H factor.

    Returns:
        list[sp.Expr]: A differential spectra balance that was created.
    """
    nonline_s = nonline.ranked()
    poly_s = poly.ranked()
    for i, d in enumerate(poly_s):
        poly_s[i] = d.subs(poly.expr_coeffs[i], poly_c[i])

    def es(i: int) -> sp.Expr:
        return (poly_s[i] if i < poly.expr_rank else 0) - (
            nonline_s[i] if i < nonline.expr_rank else 0
        )

    m = (poly.expr_rank - 1) * 2 + 1

    print("Spectrum was formed:")
    for i in np.arange(0, m):
        display(es(i))

    pre_balance: sp.Expr = sp.parse_expr("0")
    for k in np.arange(0, m):
        buff: sp.Expr = sp.parse_expr("0")
        for i in np.arange(0, k):
            buff += es(k - i) * es(i)
        buff *= 1 / (k + 1)
        pre_balance += buff
    pre_balance *= H

    print("Pre-balance was formed:")
    display(sp.collect(pre_balance, H))

    pre_balance = pre_balance.subs(H, h)

    balance: list[sp.Expr] = []
    for a in nonline.expr_coeffs:
        balance.append(sp.diff(pre_balance, a))

    return balance


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

    rr_pr = 1
    for i in np.arange(rank_range):
        data_y = poly_fit_lite(data, rank + i)

        rse = Metrics.rse(data, data_y)
        r_sq = Metrics.r_sq(data, data_y)
        if i != 0:
            rr = np.abs(rse_pr - rse) * np.abs(r_sq_pr - r_sq + 1)
            if rr > rr_pr:
                return rank + i - 1
            rr_pr = rr

        rse_pr = rse
        r_sq_pr = r_sq

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
        system[i] = eq.subs(H, 1)
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
    factors.insert(0, spectrum.var_main_sp)
    solution = sc.optimize.curve_fit(
        sp.lambdify(factors, spectrum.expr_sp),
        data_x,
        data,
        p0=solution,
    )[0]
    return solution
