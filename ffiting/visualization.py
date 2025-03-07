"""Utility functions to create generic graphs for produced data. Yes, we are
that lazy.
"""

from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt

from .common import np


@dataclass
class PlotRequest:
    """Basic data for creating graphic projections."""

    data: np.ndarray
    label: str
    label_y: str = field(default="data_y")
    label_x: str = field(default="data_x")
    xlims: Optional[tuple[float, float]] = field(default=None)
    ylims: Optional[tuple[float, float]] = field(default=None)


def multi_plot(
    *requests: PlotRequest,
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    fig_scale: int = 5,
    label_x="data_x",
    label_y="data_y",
) -> None:
    """Displays given request on singular plot, multiplexing the data over each
    other. Requests are used only for required members (data, label), everything
    else is supplied through function arguments.

    Arguments:
        data_x (np.ndarray): Vector to be used as `x` axis.
        *requests (PlotRequest): Variable number of plot creation requests. Supply
        vector for `y` axis and label for each plot.
        ylims (tuple[float, float], optional): Sets upper and lover bounds
        for the figure. Defaults to None.
        fig_scale (int, optional): Increases or decreases the resolution of generated
        graph. Used as coefficient, defaults to 5.
        label_data (str, optional): Separate name for `x` axis. Defaults to "data".
    """
    n_plots = len(requests)
    fig, ax = plt.subplots()

    for i in range(n_plots):
        ax.plot(requests[i].data, label=requests[i].label)
    ax.legend()
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    fig.set_figwidth(fig_scale)
    fig.set_figheight(fig_scale)
    plt.show()


def sep_plot(
    *requests: PlotRequest,
    row_size: int = 2,
    fig_scale: int = 5,
) -> None:
    """Displays given plot requests as separate graphs mashed together into grid.
    Uses optional fields of `PlotRequest` object.

    Args:
        *requests (PlotRequest): Arbitrary number of plot creation data, which
        is required. Provides data for `x` and `y` axis for each subplot, labels,
        as well as other configs.
        row_size (int, optional): Sets maximum subplot amount per row. Defaults to 3.
        fig_scale (int, optional): Increases or decreases the resolution of generated
        graph. Used as coefficient, defaults to 5.
    """
    n_plots = len(requests)
    n_rows = (
        int(np.ceil(n_plots / row_size))
        if int(np.floor(n_plots / row_size)) >= 1
        else 1
    )

    fig, ax = plt.subplots(n_rows, row_size)
    for i in range(n_plots):
        if ax.ndim > 1:
            row = int(np.floor(i / row_size))
            col = i - row * row_size
            ax[row, col].plot(requests[i].data, label=requests[i].label)
            ax[row, col].legend()
            ax[row, col].set_xlabel(requests[i].label_x)
            ax[row, col].set_ylabel(requests[i].label_y)
            if requests[i].xlims:
                ax[row, col].xlim(requests[i].xlims)
            if requests[i].ylims:
                ax[row, col].ylim(requests[i].ylims)
        else:
            ax[i].plot(requests[i].data, label=requests[i].label)
            ax[i].legend()
            ax[i].set_xlabel(requests[i].label_x)
            ax[i].set_ylabel(requests[i].label_y)
            if requests[i].xlims:
                ax[i].xlim(requests[i].xlims)
            if requests[i].ylims:
                ax[i].ylim(requests[i].ylims)
    fig.set_figwidth(row_size * fig_scale)
    fig.set_figheight(n_rows * fig_scale)
    plt.show()
