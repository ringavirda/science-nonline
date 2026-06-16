import numpy as np
from dataclasses import dataclass
from dtfit.infra import ModelData
from dtfit.log import logger
from .metrics import get_metrics, get_distribution

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is an optional 'viz' extra
    plt = None


def _require_plt() -> None:
    """Ensure matplotlib is available before plotting."""
    if plt is None:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install 'dtfit[viz]'"
        )


def display_data_metrics(model_data: ModelData) -> None:
    """Display the metrics for a given model data."""
    if model_data.data_fitted is None:
        logger.info("No fitted data available to calculate metrics.")
        return
    display_metrics(model_data.data_y, model_data.data_fitted)


def display_metrics(
    data_origin: np.ndarray,
    data_fitted: np.ndarray,
) -> None:
    """Display the metrics for a given data."""
    metrics = get_metrics(data_origin, data_fitted)
    print(f"Calculated Metrics:\n{metrics}")


@dataclass
class PlotOptions:
    title: str = "Model Data"
    xlabel: str = "x"
    ylabel: str = "f(x)"
    origin: bool = True
    noisy: bool = True
    poly: bool = True
    nonline: bool = True


def display_data_plot(
    model_data: ModelData, options: PlotOptions = PlotOptions()
) -> None:
    """Display the plot for a given model data."""
    _require_plt()
    plt.figure(figsize=(10, 6))
    if options.origin and model_data.data_y is not None:
        plt.plot(model_data.data_x, model_data.data_y, label="Original")
    if options.poly and model_data.data_fitted is not None:
        plt.plot(
            model_data.data_x,
            model_data.data_fitted,
            label="Fitted",
        )

    plt.legend()
    plt.title(options.title)
    plt.xlabel(options.xlabel)
    plt.ylabel(options.ylabel)
    plt.grid()
    plt.show()


def display_plot(
    data_x: np.ndarray,
    *data_display: tuple[np.ndarray, str] | np.ndarray,
    options: PlotOptions = PlotOptions(),
) -> None:
    """Display a plot of the provided data arrays."""
    _require_plt()
    plt.figure(figsize=(10, 6))
    if len(data_display) < 1:
        raise ValueError("At least one data array must be provided.")

    for i, item in enumerate(data_display):
        if isinstance(item, tuple):
            data, label = item
            plt.plot(data_x, data, label=label)
        else:
            plt.plot(data_x, item, label=f"Data {i+1}")  # type: ignore

    plt.legend()
    plt.title(options.title)
    plt.xlabel(options.xlabel)
    plt.ylabel(options.ylabel)
    plt.grid()
    plt.show()


def display_plot_distribution(
    data_origin: np.ndarray,
    data_fitted: np.ndarray,
    options: PlotOptions = PlotOptions(
        xlabel="Residuals", ylabel="Density", title="Residuals Distribution"
    ),
) -> None:
    """Display a plot of the distribution of the provided data."""
    _require_plt()
    distribution = get_distribution(data_fitted - data_origin)
    plt.figure(figsize=(10, 6))
    plt.hist(
        data_fitted - data_origin, bins=30, density=True, alpha=0.6, color="g"
    )
    x = np.linspace(
        min(data_fitted - data_origin), max(data_fitted - data_origin), 100
    )
    plt.plot(
        x,
        distribution.pdf(x),
        label=f"Distribution: {distribution.name}",
        color="blue",
    )
    plt.legend()
    plt.title(options.title)
    plt.xlabel(options.xlabel)
    plt.ylabel(options.ylabel)
    plt.grid()
    plt.show()
