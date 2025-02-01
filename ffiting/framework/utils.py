""" Additional utilities for statistics and data manipulation.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Self, Tuple
from time import perf_counter
from scipy.interpolate import interp1d

import ffiting.framework as fr
from ..common import np


@dataclass(frozen=True)
class ModelMetrics:
    """Readonly container for main model stats, that can be used to evaluate
    fitting effectiveness.
    """

    rse: float  # Residual Squared Error
    mse: float  # Mean Squared Error
    lin_div: float  # Linear Deviation
    std_div: float  # Standard Deviation
    std_err: float  # Standard Error
    r_sq: float  # Determination Coefficient
    corr: float  # Correlation Coefficient
    ccord: float  # Concordance Coefficient


class Metrics:
    """Container for most available operations to calculate metrics for data and
    models. Methods from here are used for `ModelMetrics`.
    """

    @staticmethod
    def period(data: np.ndarray) -> int:
        """Calculates possible period length. If it returns values bigger than
        input array size the data is probably not periodic.
        """
        position = 0
        origin = data[position]
        passes = 0
        for n in np.nditer(data):
            if n == origin:
                passes += 1
            elif passes == 2:
                return position
            else:
                position += 1

    @staticmethod
    def growth(data: np.ndarray) -> float:
        """Returns average jump in value between all points in the given vector.
        If value is positive, than data generally ascends, if not - descends.
        """
        gr = np.empty(data.size - 1)
        for i in np.arange(data.size):
            if i < data.size - 1:
                gr[i] = data[i + 1] - data[i]
        return gr.mean()

    @staticmethod
    def tss(data: np.ndarray) -> float:
        """Returns the Total Sum of Squares for given vector.

        Formula:
            n := data.size
            tss := sum(0, n, i => pow(mean(data) - data[i], 2))
        """
        res = 0.0
        mean = data.mean()
        for y in np.nditer(data):
            res += np.square(y - mean)
        return res

    @staticmethod
    def timed(
        proc: Callable[(...), Any], label: str = None, echo: bool = False
    ) -> Tuple[Any, float]:
        """Runs performance counter before and after procedure call. Returns the
        result of the operation and time difference packed into a cortege. Can
        print time taken to std output if flag is set.
        """
        time = perf_counter()
        res = proc()
        time = perf_counter() - time
        label = proc.__name__ if not label else label
        if echo:
            print(f"[{label}] took {time*1000:.5f}ms to complete.")
        return (res, time)

    @staticmethod
    def __size_equal(
        func: Callable[[np.ndarray, np.ndarray], Any]
    ) -> Callable[[np.ndarray, np.ndarray], Any]:
        """Internal class decorator to guard against the use of methods on
        inappropriate data vectors. Fails if those have different size.
        """

        def inner(left: np.ndarray, right: np.ndarray) -> Any:
            if left.size != right.size:
                raise RuntimeError("Array sizes must be equal.")
            return func(left, right)

        return inner

    @staticmethod
    @__size_equal
    def lin_div(left: np.ndarray, right: np.ndarray) -> float:
        """Returns Linear Deviation between given data vectors.

        Formula:
            n := left.size = right.size
            lin_div := sum(0, n, i => abs(left[i] - right[i]))
        """
        div = 0.0
        for i, v in np.ndenumerate(left):
            div += np.abs(v - right[i])
        return div

    @staticmethod
    @__size_equal
    def concord(left: np.ndarray, right: np.ndarray) -> float:
        """Returns Concordance coefficient for given vectors.

        Formula:
            concord := abs(2 * cov(left, right) /
                (var(left) + var(right) + pow(median(left) - median(right), 2)))
        """
        return np.abs(
            2
            * np.cov(left, right, bias=True)[0][1]
            / (
                np.var(left)
                + np.var(right)
                + np.power(np.median(left) - np.median(right), 2)
            )
        )

    @staticmethod
    @__size_equal
    def rss(left: np.ndarray, right: np.ndarray) -> float:
        """Returns Residual Squared Sum for given data.

        Formula:
            n := left.size = right.size
            rss := sum(0, n, i => pow(left[i] - right[i], 2))
        """
        res = 0.0
        for i, l in np.ndenumerate(left):
            res += np.square(l - right[i])
        return res

    @staticmethod
    @__size_equal
    def mse(left: np.ndarray, right: np.ndarray) -> float:
        """Returns Mean Squared Error for given data vectors.

        Formula:
            n := left.size = right.size
            mse := mean(sum(0, n, i => pow(left[i] - right[i], 2)))
        """
        return (np.square(left - right)).mean()

    @classmethod
    @__size_equal
    def rse(cls: Self, left: np.ndarray, right: np.ndarray) -> float:
        """Returns Residual Squared Error for given data vectors. Uses the `rss`
        metric internally.

        Formula:
            n := left.size = right.size
            rse := sqrt(rss(left, right) / n - 2)
        """
        return np.sqrt(cls.rss(left, right) / (left.size - 2))

    @classmethod
    @__size_equal
    def r_sq(cls: Self, left: np.ndarray, right: np.ndarray) -> float:
        """Returns Determination Coefficient (R Squared) for given data vectors.
        Uses `rss` and `tss` metrics internally.

        Formula:
            r_sq := 1 - rss(left, right) / tss(right)
        """
        return 1 - cls.rss(left, right) / cls.tss(right)


def get_metrics(model: fr.Model | fr.ModelLite) -> ModelMetrics:
    """Calculates most used metrics for given model. Uses implementations from
    `Metrics` container and from `numpy` directly.

    Arguments:
        model (Model | ModelLite): An instance to perform calculations on.

    Returns:
        ModelMetrics: Data object with fields populated using data from given model.
    """
    return ModelMetrics(
        rse=Metrics.rse(model.data_raw, model.data_fit),
        mse=Metrics.mse(model.data_raw, model.data_fit),
        r_sq=Metrics.r_sq(model.data_raw, model.data_fit),
        lin_div=Metrics.lin_div(model.data_raw, model.data_fit),
        std_div=np.std(model.data_raw, model.data_fit),
        std_err=np.std(model.data_raw, model.data_fit) / np.sqrt(model.data_raw.size),
        corr=np.corrcoef(model.data_raw, model.data_fit)[0, 1],
        ccord=Metrics.concord(model.data_raw, model.data_fit),
    )


def scale_data(data: np.ndarray, coeff: float) -> np.ndarray:
    """Generates new dataset from the given one through resizing it. Implements
    interpolation to infer new points in the data, so it may have some minor error
    init. It does not extend the radius of given data, simply increases or decreases
    the amount of points in the vector.

    Arguments:
        data (ndarray): Vector to be scaled.
        coeff (float): A value to represent by how much the data needs to be resized.
        If it is set to 2 than vector with double the size returns, if 0.5 - half,
        if 1 - no changes.

    Returns:
        ndarray: New generated scaled vector from input data.
    """
    range_new = np.arange(0, data.size, 1 / coeff)
    inter = interp1d(np.arange(data.size), data)
    return inter(range_new)


def collapse(
    func: Callable[[float], float]
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Wrapper for generation or modelling functions that cannot automatically parse arrays."""

    def inner(value: float | np.ndarray) -> float | np.ndarray:
        if isinstance(value, float):
            return func(value)
        elif isinstance(value, np.ndarray):
            res = np.empty(value.shape)
            for i, v in np.ndenumerate(value):
                res[i] = func(v)
            return res

    return inner


@dataclass(frozen=True)
class NoiseConfig:
    """Contains configuration necessary for noise generation in data generators."""

    mu: float = field(default=0.0)
    sigma: float = field(default=5.0)
    abnormals: bool = field(default=False)
    coeff: float = field(default=3.0)
    density: float = field(default=10.0)


def apply_noise(data: float | np.ndarray, config=NoiseConfig()) -> float | np.ndarray:
    """Configurable noise generator to simulate absolute error in generated datasets."""
    if isinstance(data, float):
        error = np.random.normal(config.mu, config.sigma, 1)
        return data + error[0]
    elif isinstance(data, np.ndarray):
        # Normal deviations
        polluted = np.empty(data.shape)
        errors = np.random.normal(config.mu, config.sigma, data.size)
        with np.nditer(polluted, op_flags=["readwrite"], flags=["f_index"]) as it:
            for value in it:
                value[...] += data[it.index] + errors[it.index]
        if config.abnormals:
            # Abnormal deviations
            abnormal_count = int((data.size * config.density) / 100)
            abnormal_pos = np.zeros(abnormal_count)
            # Fill in positions using normal distribution
            with np.nditer(abnormal_pos, op_flags=["readwrite"]) as it:
                for value in it:
                    value[...] = np.ceil(np.random.randint(0, data.size))
            # Fill in abnormals
            abnormals = np.random.normal(
                config.mu, config.sigma * config.coeff, abnormal_count
            )
            with np.nditer(
                abnormal_pos, op_flags=["readwrite"], flags=["f_index"]
            ) as it:
                for value in it:
                    polluted[int(value)] += abnormals[it.index]
        return polluted
    else:
        raise ValueError("Noise can only be applied to float or array values.")
