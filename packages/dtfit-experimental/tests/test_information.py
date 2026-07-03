"""InformationFilter -- the info-form fusion primitive (demoted from stable dtfit
to the experimental tier: it is coherent and tested but exercised by no domain
study, so it has not cleared the >=2-domain promotion gate)."""

import numpy as np

from dtfit_experimental import InformationFilter


def test_information_filter_matches_ols_and_fuses_exactly():
    """Info-form estimator matches OLS (the covariance-form answer), a vector
    measurement equals the per-row loop, and fusing two independent estimators is
    exact and order-independent (information is additive)."""
    rng = np.random.default_rng(0)
    n, m = 3, 400
    theta = np.array([1.5, -0.7, 2.0])
    H = rng.standard_normal((m, n))
    z = H @ theta + 0.1 * rng.standard_normal(m)

    f = InformationFilter(n, prior_precision=1e-9)
    for i in range(m):
        f.partial_fit(H[i], z[i], r=1.0)
    ols = np.linalg.lstsq(H, z, rcond=None)[0]
    assert np.allclose(f.theta_, ols, atol=1e-8)

    fv = InformationFilter(n, prior_precision=1e-9)
    fv.partial_fit(H, z, r=1.0)                      # one vector call
    assert np.allclose(fv.theta_, f.theta_, atol=1e-10)

    a = InformationFilter(n, prior_precision=1e-9)
    a.partial_fit(H[:150], z[:150])
    b = InformationFilter(n, prior_precision=1e-9)
    b.partial_fit(H[150:], z[150:])
    # fuse in both orders -> same as processing the whole stream
    ab = InformationFilter(n, prior_precision=1e-9)
    ab.partial_fit(H[:150], z[:150])
    ab.fuse(b)
    ba = InformationFilter(n, prior_precision=1e-9)
    ba.partial_fit(H[150:], z[150:])
    ba.fuse(a)
    assert np.allclose(ab.theta_, f.theta_, atol=1e-9)
    assert np.allclose(ba.theta_, f.theta_, atol=1e-9)
    # cov readout is the small inverse
    assert f.cov_.shape == (n, n)
