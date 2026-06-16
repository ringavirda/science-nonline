from time import perf_counter
import numpy as np
from typing import Any, Callable, Tuple


def period(data: np.ndarray) -> int:
    """
    Calculates possible period length. If it returns values bigger than
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
    return 0


def growth(data: np.ndarray) -> float:
    """
    Returns average jump in value between all points in the given vector.
    If value is positive, than data generally ascends, if not - descends.
    """
    gr = np.empty(data.size - 1)
    for i in np.arange(data.size):
        if i < data.size - 1:
            gr[i] = data[i + 1] - data[i]
    return gr.mean()


def timed(
    proc: Callable[(...), Any], label: str | None = None, echo: bool = False
) -> Tuple[Any, float]:
    """
    Runs performance counter before and after procedure call. Returns the
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
