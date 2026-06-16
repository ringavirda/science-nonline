from scipy.optimize import least_squares

import numpy as np
import sympy as sp


def solve_nonline(
    system: list[sp.Expr | sp.Basic] | sp.Expr,
    coeffs: list[sp.Symbol],
) -> list[list[sp.Expr | sp.Basic]]:
    """
    Solve given system of equations using old sympy solver. This may find multiple
    improper solutions, so it filters out all of the complex and incomplete ones.

    Args:
        system (list[sp.Expr | sp.Basic]): A system of expressions to solve.
        coeffs (list[sp.Symbol]): List of coefficients to solve system around.

    Raises:
        RuntimeError: If no solutions were found.

    Returns:
        list[sp.Expr | sp.Basic]: List of found solutions.
    """
    solutions0 = sp.nonlinsolve(system, coeffs)

    solutions: list[list[sp.Expr | sp.Basic]] = []
    for solution in solutions0.args:
        solution = [sp.re(sp.N(val, chop=True)) for val in solution]  # type: ignore
        solutions.append(solution)  # type: ignore
    
    if len(solutions) == 0:
        raise RuntimeError(
            "No suitable solutions were found during nonline solve!"
        )
    
    # Add metrics filtering
    for solution in solutions:
        for val in solution:
            if val == 0.0 or val == 0 or sp.im(val) != 0:
                solutions.remove(solution)
                break

    if len(solutions) == 0:
        raise RuntimeError(
            "No suitable solutions were found during nonline solve!"
        )

    return solutions


def solve_numeric(
    system: list[sp.Expr | sp.Basic] | sp.Expr,
    coeffs: list[sp.Symbol],
    solution0: np.ndarray,
) -> np.ndarray:
    """
    Uses nonlinear least squares to solve given system of equations.
    Requires starting values.

    Arguments:
        system (list[sp.Expr | sp.Basic]): A list of sympy expressions 
        representing the system of equations to solve. 
        coeffs (list[sp.Symbol]): A list of sympy symbolsrepresenting the 
        coefficients to solve for. 
        solution0 (np.ndarray): An initial guess for the solution.
    Returns:
        np.ndarray: Array of the solution that was optimized using a possibly
        overdefined system of equations.
    """
    solution0 = solution0.astype(np.float64)
    func = sp.lambdify(coeffs, system, "scipy")

    def wrapper(args: np.ndarray) -> np.float64:
        return func(*args)

    solution = least_squares(wrapper, solution0)

    return np.array(solution.x).astype(np.float64)
