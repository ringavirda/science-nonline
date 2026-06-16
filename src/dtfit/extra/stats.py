import numpy as np
from typing import Any, Callable


def size_equals(
    func: Callable[[np.ndarray, np.ndarray], Any],
) -> Callable[[np.ndarray, np.ndarray], Any]:
    """
    Decorator to guard against the use of methods on inappropriate data vectors.
    Fails if those have different size.
    """

    def inner(left: np.ndarray, right: np.ndarray) -> Any:
        if left.size != right.size:
            raise RuntimeError("Array sizes must be equal.")
        return func(left, right)

    return inner


def tss(data: np.ndarray) -> float:
    """
    Returns the Total Sum of Squares for given vector.

    Formula:
        n := data.size
        tss := sum(0, n, i => pow(mean(data) - data[i], 2))
    """
    res = 0.0
    mean = data.mean()
    for y in np.nditer(data):
        res += np.square(y - mean)
    return res


@size_equals
def lin_div(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Linear Deviation between given data vectors.

    Formula:
        n := left.size = right.size
        lin_div := sum(0, n, i => abs(left[i] - right[i]))
    """
    div = 0.0
    for i, v in np.ndenumerate(left):
        div += np.abs(v - right[i])
    return div


@size_equals
def rss(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Residual Squared Sum for given data.

    Formula:
        n := left.size = right.size
        rss := sum(0, n, i => pow(left[i] - right[i], 2))
    """
    res = 0.0
    for i, l in np.ndenumerate(left):
        res += np.square(l - right[i])
    return res


@size_equals
def mse(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Mean Squared Error for given data vectors.

    Formula:
        n := left.size = right.size
        mse := mean(sum(0, n, i => pow(left[i] - right[i], 2)))
    """
    return (np.square(left - right)).mean()


@size_equals
def rse(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Residual Squared Error for given data vectors. Uses the `rss`
    metric internally.

    Formula:
        n := left.size = right.size
        rse := sqrt(rss(left, right) / n - 2)
    """
    if left.size != right.size:
        raise RuntimeError("Array sizes must be equal.")
    return np.sqrt(rss(left, right) / (left.size - 2))


@size_equals
def r_sq(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Determination Coefficient (R Squared) for given data vectors.
    Uses `rss` and `tss` metrics internally.

    Formula:
        r_sq := 1 - rss(left, right) / tss(right)
    """
    if left.size != right.size:
        raise RuntimeError("Array sizes must be equal.")
    return 1 - rss(left, right) / tss(right)


@size_equals
def std_div(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Standard Deviation between two data vectors.

    Formula:
        n := left.size = right.size
        std_div := sqrt(sum(0, n, i => pow(left[i] - right[i], 2)) / n)
    """
    res = 0.0
    for i, l in np.ndenumerate(left):
        res += np.square(l - right[i])
    return np.sqrt(res / left.size)


@size_equals
def std_err(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Standard Error between two data vectors.

    Formula:
        n := left.size = right.size
        std_err := std_div(left, right) / sqrt(n)
    """
    return std_div(left, right) / np.sqrt(left.size)


@size_equals
def corr(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Correlation Coefficient between two data vectors.

    Formula:
        corr := cov(left, right) / (std_dev(left) * std_dev(right))
    """
    return np.corrcoef(left, right)[0, 1]


@size_equals
def ccord(left: np.ndarray, right: np.ndarray) -> float:
    """
    Returns Concordance Coefficient between two data vectors.

    Formula:
        ccord := abs(2 * cov(left, right) * var(left) * var(right) /
            (var(left)^2 + var(right)^2 + (mean(left) - mean(right)^2))
    """
    return np.abs(
        2
        * np.cov(left, right, bias=True)[0][1]
        / (
            np.var(left)
            + np.var(right)
            + np.square(np.median(left) - np.median(right))
        )
    )
