"""dtfit.simulation -- synthetic data tooling for experiments and benchmarks.

This subpackage is intentionally NOT re-exported from the top-level ``dtfit``
namespace: end users fit their own data and do not need synthetic generators
or noise injection. It exists to reproduce the dissertation's experiments and
to benchmark methods on controlled inputs.

    from dtfit.simulation import gen_exponential4, apply_normal_noise
    from dtfit.simulation import DataGenerationMw, DataPollutionMw
"""

from .generators import (
    gen_combined_lint,
    gen_exponential4,
    gen_ranked_poly,
    gen_transcendental,
)
from .noise import (
    apply_abnormal_noise,
    apply_normal_noise,
    apply_uniform_noise,
    apply_uniform_abnormal_noise,
)
from .generation import DataGenerationMw
from .pollution import DataPollutionMw, NoiseConfig, AbnormalsConfig
from .expressions import Prefabs

__all__ = [
    "gen_combined_lint",
    "gen_exponential4",
    "gen_ranked_poly",
    "gen_transcendental",
    "apply_normal_noise",
    "apply_uniform_noise",
    "apply_abnormal_noise",
    "apply_uniform_abnormal_noise",
    "DataGenerationMw",
    "DataPollutionMw",
    "NoiseConfig",
    "AbnormalsConfig",
    "Prefabs",
]
