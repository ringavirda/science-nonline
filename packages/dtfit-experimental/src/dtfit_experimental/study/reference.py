"""Certified reference data: the NIST StRD nonlinear-regression corpus and
the vendored Puromycin enzyme-kinetics dataset.

The StRD files carry NIST's own certified parameter values, so a fit against
them is checked against ground truth rather than against another fit. The
corpus itself is fetched by :func:`dtfit_experimental.study.download_data.nist`
into ``data_dir()/"nist"``; :func:`load` reads it from there and raises when it
is absent. Puromycin has no such fetch step and is vendored in this module as
literal arrays instead (see :func:`puromycin`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .paths import data_dir

# Model expression (sympy syntax, in x and the b-parameters) and NIST's
# stated difficulty level, one entry per dataset this module ships. Fixed by
# the plan: the three difficulty levels, the two datasets the papers already
# quote (Misra1a, Rat42) and the two where NLLS diverges from Start 1
# (MGH17, BoxBOD).
_MODELS: dict[str, tuple[str, str]] = {
    "Misra1a": ("b1*(1 - exp(-b2*x))", "lower"),
    "DanWood": ("b1*x**b2", "lower"),
    "Chwirut2": ("exp(-b1*x)/(b2 + b3*x)", "lower"),
    "Misra1c": ("b1*(1 - 1/sqrt(1 + 2*b2*x))", "average"),
    "Roszman1": ("b1 - b2*x - atan(b3/(x - b4))/pi", "average"),
    "MGH17": ("b1 + b2*exp(-x*b4) + b3*exp(-x*b5)", "average"),
    "Rat42": ("b1/(1 + exp(b2 - b3*x))", "higher"),
    "BoxBOD": ("b1*(1 - exp(-b2*x))", "higher"),
    "Eckerle4": ("(b1/b2)*exp(-((x - b3)**2)/(2*b2**2))", "higher"),
    "MGH10": ("b1*exp(b2/(x + b3))", "higher"),
}


@dataclass(frozen=True)
class NistDataset:
    """One StRD nonlinear-regression dataset with its certified answer.

    Attributes:
        name: StRD dataset name, e.g. ``"Rat42"``.
        level: NIST's stated difficulty: ``"lower"``, ``"average"`` or
            ``"higher"``.
        expr: The model as a sympy expression in ``x`` and the parameters
            named in ``names``.
        names: Parameter names in NIST's own order, e.g. ``["b1", "b2"]``.
            ``dtfit.fit`` on a string model does not honour this order (it
            fits in sorted name order); a caller seeds and reads by name.
        start1: Far starting values, one per name, NIST's "Start 1".
        start2: Near starting values, one per name, NIST's "Start 2".
        certified: Certified parameter values, one per name; the reference
            truth, not a fit.
        certified_sd: Certified parameter standard deviations, one per
            name, in the parameter's own units.
        residual_sum_of_squares: Certified residual sum of squares at the
            certified parameters, in the response variable's squared units.
        x: Predictor values, one per observation.
        y: Response values, one per observation.
    """

    name: str
    level: str
    expr: sp.Expr
    names: list[str]
    start1: np.ndarray
    start2: np.ndarray
    certified: np.ndarray
    certified_sd: np.ndarray
    residual_sum_of_squares: float
    x: np.ndarray
    y: np.ndarray


def datasets() -> list[tuple[str, str]]:
    """The ten certified StRD datasets this module ships.

    Returns:
        ``(name, level)`` pairs in the fixed order declared above (not
        NIST's own catalogue order), ``level`` one of ``"lower"``,
        ``"average"``, ``"higher"``.
    """
    return [(name, level) for name, (_, level) in _MODELS.items()]


def _span(text: str, label: str) -> tuple[int, int]:
    """1-based inclusive (start, end) line numbers of a StRD header block.

    Reads the span from the file's own ``"<label>   (lines A to B)"`` header
    line rather than a fixed offset, since the block boundaries move with
    the parameter count.
    """
    m = re.search(rf"{label}\s+\(lines\s+(\d+)\s+to\s+(\d+)\)", text)
    if not m:
        raise ValueError(f"no {label!r} line span in this StRD file")
    return int(m.group(1)), int(m.group(2))


def _parse(text: str, name: str, expr_str: str, level: str) -> NistDataset:
    lines = text.splitlines()
    a, b = _span(text, "Starting Values")
    names, s1, s2, cert, cert_sd = [], [], [], [], []
    for ln in lines[a - 1 : b]:
        parts = ln.replace("=", " ").split()
        if len(parts) < 5:
            continue
        names.append(parts[0])
        s1.append(float(parts[1]))
        s2.append(float(parts[2]))
        cert.append(float(parts[3]))
        cert_sd.append(float(parts[4]))
    m = re.search(r"Residual Sum of Squares:\s+([0-9.eE+-]+)", text)
    if not m:
        raise ValueError("no 'Residual Sum of Squares' line in this StRD file")
    rss = float(m.group(1))
    a, b = _span(text, "Data")
    rows = [ln.split() for ln in lines[a - 1 : b] if ln.strip()]
    y = np.array([float(r[0]) for r in rows])
    x = np.array([float(r[1]) for r in rows])
    return NistDataset(
        name=name,
        level=level,
        expr=sp.sympify(expr_str),
        names=names,
        start1=np.array(s1),
        start2=np.array(s2),
        certified=np.array(cert),
        certified_sd=np.array(cert_sd),
        residual_sum_of_squares=rss,
        x=x,
        y=y,
    )


def load(name: str) -> NistDataset:
    """Load one certified StRD dataset by name.

    Args:
        name: One of the names :func:`datasets` lists, e.g. ``"Rat42"``.

    Returns:
        Its :class:`NistDataset`, parsed from
        ``data_dir()/"nist"/f"{name}.dat"``.

    Raises:
        KeyError: ``name`` is not one of the ten catalogued datasets.
        FileNotFoundError: the corpus has not been downloaded; names the
            expected path. Run
            ``python -m dtfit_experimental.study.download_data`` to fetch it.
        ValueError: the file is present but malformed: no "Starting Values"
            or "Data" line span, or no "Residual Sum of Squares" line.
    """
    expr_str, level = _MODELS[name]
    path = data_dir() / "nist" / f"{name}.dat"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run "
            "`python -m dtfit_experimental.study.download_data` to fetch the "
            "NIST StRD corpus"
        )
    return _parse(path.read_text(), name, expr_str, level)


# Bates, D.M. and Watts, D.G. (1988). Nonlinear Regression Analysis and Its
# Applications. Wiley. The Puromycin enzyme-kinetics data, treated group:
# initial reaction velocity (counts/min/min) of an enzymatic reaction against
# substrate concentration (ppm), six concentrations in replicate pairs.
_PUROMYCIN_CONC = (
    0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10,
)
_PUROMYCIN_VELOCITY = (
    76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200,
)


def puromycin() -> tuple[np.ndarray, np.ndarray]:
    """The vendored Puromycin dataset (Bates and Watts 1988, treated group).

    Returns:
        ``(conc, velocity)``, twelve pairs as float arrays: ``conc`` the
        substrate concentration in ppm, ``velocity`` the initial reaction
        velocity in counts/min/min.
    """
    return np.array(_PUROMYCIN_CONC, dtype=float), np.array(
        _PUROMYCIN_VELOCITY, dtype=float
    )
