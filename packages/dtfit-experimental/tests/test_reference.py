"""Certified reference data: the NIST StRD parser and the vendored Puromycin
dataset.

Each test names the mutation of reference.py it fails under.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from dtfit_experimental.study import reference
from dtfit_experimental.study.paths import data_dir

# Verbatim NIST StRD file (Misra1a.dat): 2 parameters, so the Starting Values
# and Data blocks sit at their usual offsets.
_MISRA1A = """\
NIST/ITL StRD
Dataset Name:  Misra1a           (Misra1a.dat)

File Format:   ASCII
               Starting Values   (lines 41 to 42)
               Certified Values  (lines 41 to 47)
               Data              (lines 61 to 74)

Procedure:     Nonlinear Least Squares Regression

Description:   These data are the result of a NIST study regarding
               dental research in monomolecular adsorption.  The
               response variable is volume, and the predictor
               variable is pressure.

Reference:     Misra, D., NIST (1978).
               Dental Research Monomolecular Adsorption Study.







Data:          1 Response Variable  (y = volume)
               1 Predictor Variable (x = pressure)
               14 Observations
               Lower Level of Difficulty
               Observed Data

Model:         Exponential Class
               2 Parameters (b1 and b2)

               y = b1*(1-exp[-b2*x])  +  e



          Starting values                  Certified Values

        Start 1     Start 2           Parameter     Standard Deviation
  b1 =   500         250           2.3894212918E+02  2.7070075241E+00
  b2 =     0.0001      0.0005      5.5015643181E-04  7.2668688436E-06

Residual Sum of Squares:                    1.2455138894E-01
Residual Standard Deviation:                1.0187876330E-01
Degrees of Freedom:                                12
Number of Observations:                            14












Data:   y               x
      10.07E0      77.6E0
      14.73E0     114.9E0
      17.94E0     141.1E0
      23.93E0     190.8E0
      29.61E0     239.9E0
      35.18E0     289.0E0
      40.02E0     332.8E0
      44.82E0     378.4E0
      50.76E0     434.8E0
      55.05E0     477.3E0
      61.01E0     536.8E0
      66.40E0     593.1E0
      75.47E0     689.1E0
      81.78E0     760.0E0
"""

# Verbatim NIST StRD file (Chwirut2.dat): 3 parameters, so the Certified
# Values block is longer and the header spells "lines 41 to  43" with two
# spaces - the parameter count has moved the Data block down from where a
# fixed offset (tuned against a 2-parameter file) would look for it.
_CHWIRUT2 = """\
NIST/ITL StRD
Dataset Name:  Chwirut2          (Chwirut2.dat)

File Format:   ASCII
               Starting Values   (lines 41 to  43)
               Certified Values  (lines 41 to  48)
               Data              (lines 61 to 114)

Procedure:     Nonlinear Least Squares Regression

Description:   These data are the result of a NIST study involving
               ultrasonic calibration.  The response variable is
               ultrasonic response, and the predictor variable is
               metal distance.



Reference:     Chwirut, D., NIST (197?).
               Ultrasonic Reference Block Study.





Data:          1 Response  (y = ultrasonic response)
               1 Predictor (x = metal distance)
               54 Observations
               Lower Level of Difficulty
               Observed Data

Model:         Exponential Class
               3 Parameters (b1 to b3)

               y = exp(-b1*x)/(b2+b3*x)  +  e



          Starting values                  Certified Values

        Start 1     Start 2           Parameter     Standard Deviation
  b1 =   0.1         0.15          1.6657666537E-01  3.8303286810E-02
  b2 =   0.01        0.008         5.1653291286E-03  6.6621605126E-04
  b3 =   0.02        0.010         1.2150007096E-02  1.5304234767E-03

Residual Sum of Squares:                    5.1304802941E+02
Residual Standard Deviation:                3.1717133040E+00
Degrees of Freedom:                                51
Number of Observations:                            54











Data:  y             x
      92.9000E0     0.500E0
      57.1000E0     1.000E0
      31.0500E0     1.750E0
"""


def test_parse_misra1a_reads_data_columns_in_file_order():
    d = reference._parse(_MISRA1A, "Misra1a", "b1*(1 - exp(-b2*x))", "lower")
    assert d.x.size == 14
    # fails if the y and x data columns were read swapped
    assert d.x[0] == pytest.approx(77.6)
    assert d.y[0] == pytest.approx(10.07)
    # fails if the Start 1 and Start 2 columns were read swapped
    assert d.start1 == pytest.approx([500.0, 1e-4])
    assert d.start2 == pytest.approx([250.0, 5e-4])
    assert d.certified == pytest.approx([238.94212918, 5.5015643181e-4])


def test_header_spans_read_from_header_not_a_fixed_offset():
    d = reference._parse(_CHWIRUT2, "Chwirut2", "exp(-b1*x)/(b2 + b3*x)", "lower")
    # a fixed offset tuned to Misra1a's 2-parameter layout would read the
    # Certified Values block, one parameter row short, as Starting Values
    assert d.names == ["b1", "b2", "b3"]
    assert d.start1 == pytest.approx([0.1, 0.01, 0.02])
    assert d.certified[0] == pytest.approx(1.6657666537e-1)
    assert d.residual_sum_of_squares == pytest.approx(5.1304802941e2)
    # the Data block, three rows here, sits past the longer header
    assert d.x.size == 3
    assert d.x[0] == pytest.approx(0.5)
    assert d.y[0] == pytest.approx(92.9)


# Fails if _span stops raising on a header with no matching line span
# (for example, a search that falls back to a fixed offset instead).
def test_parse_raises_on_missing_line_span():
    with pytest.raises(ValueError, match="no 'Starting Values' line span"):
        reference._parse("no header lines at all", "X", "b1*x", "lower")


# Fails if _parse defaults the residual sum of squares instead of raising
# when its line is absent.
def test_parse_raises_on_missing_residual_sum_of_squares():
    text = (
        "Starting Values   (lines 3 to 3)\n\n\n"
        "b1  1 2 3 4 5\n"
        "no rss line here\n"
        "Data   (lines 6 to 6)\n\n\n\n\n"
        "1.0 2.0\n"
    )
    with pytest.raises(ValueError, match="no 'Residual Sum of Squares' line"):
        reference._parse(text, "X", "b1*x", "lower")


def test_puromycin_returns_twelve_pairs_in_file_order():
    conc, velocity = reference.puromycin()
    assert conc.size == 12
    assert velocity.size == 12
    # fails if the conc and velocity columns were vendored transposed
    assert conc.min() == pytest.approx(0.02)
    assert velocity.max() == pytest.approx(207.0)


@pytest.mark.skipif(
    not (data_dir() / "nist").is_dir(), reason="NIST StRD corpus not downloaded"
)
def test_nist_corpus_fits_within_tolerance_of_certified():
    # fails on any mistyped model expression: NLLS from the near start would
    # not land within a hundredth of a percent of the certified values
    for name, _level in reference.datasets():
        d = reference.load(name)
        f = sp_lambdify(d)
        p, _ = curve_fit(f, d.x, d.y, p0=list(d.start2), maxfev=20000)
        err = np.max(np.abs(np.asarray(p) - d.certified) / np.abs(d.certified)) * 100
        assert err < 1e-2, f"{name}: {err:.3g}% off certified"


def sp_lambdify(d: reference.NistDataset):
    import sympy as sp

    syms = sp.symbols(["x", *d.names])
    f = sp.lambdify(syms, d.expr, "numpy")

    def func(xx, *p, _f=f):
        return _f(xx, *p)

    return func
