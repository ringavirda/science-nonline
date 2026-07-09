"""Full scikit-learn estimator-check suite for ``NonlineRegressor``.

The estimator declares ``input_tags.one_d_array`` (like sklearn's own
``IsotonicRegression``), so the suite exercises it with 1-D X. A handful of
check *bodies* cannot run against 1-D X at all -- they index ``X[:, 0]`` or
``X.shape[1]`` on the X they just made 1-D and crash before ever reaching the
estimator -- and one check contradicts the deliberate 1-D-input API. Those are
declared below with per-check reasons; everything else must pass.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from dtfit import NonlineRegressor

# The check body itself slices ``X[:, 0]`` (to build y) on the 1-D X that the
# ``one_d_array`` tag requests, raising IndexError before the estimator runs.
_BODY_SLICES_1D_X = (
    "check body indexes X[:, 0] on the 1-D X requested by the one_d_array tag"
)
# The check body asserts against ``X.shape[1]``, which a 1-D X does not have.
_BODY_READS_SHAPE1 = (
    "check body reads X.shape[1] on the 1-D X requested by the one_d_array tag"
)
# The sample_weight check builds a fixed multi-feature X without honoring the
# one_d_array tag, so the single-feature estimator rejects it as multi-feature
# before the sample_weight logic under test is ever reached.
_SW_BODY_MULTIFEATURE_X = (
    "check body builds a multi-feature X (it does not reshape for the "
    "one_d_array tag), which the single-feature estimator rejects before the "
    "sample_weight handling under test runs"
)


def _expected_failed_checks(estimator: NonlineRegressor) -> dict[str, str]:
    return {
        "check_dont_overwrite_parameters": _BODY_SLICES_1D_X,
        "check_dict_unchanged": _BODY_SLICES_1D_X,
        "check_f_contiguous_array_estimator": _BODY_SLICES_1D_X,
        "check_methods_sample_order_invariance": _BODY_SLICES_1D_X,
        "check_methods_subset_invariance": _BODY_SLICES_1D_X,
        "check_fit2d_1sample": _BODY_SLICES_1D_X,
        "check_fit2d_1feature": _BODY_SLICES_1D_X,
        "check_fit2d_predict1d": _BODY_SLICES_1D_X,
        "check_regressors_no_decision_function": _BODY_SLICES_1D_X,
        "check_n_features_in": _BODY_READS_SHAPE1,
        "check_n_features_in_after_fitting": _BODY_READS_SHAPE1,
        # sample_weight checks that build a multi-feature X without honoring the
        # one_d_array tag (so the single-feature estimator rejects it first).
        "check_sample_weights_shape": _SW_BODY_MULTIFEATURE_X,
        "check_sample_weights_not_overwritten": _SW_BODY_MULTIFEATURE_X,
        "check_sample_weight_equivalence_on_dense_data": (
            _SW_BODY_MULTIFEATURE_X
            + "; and for an integral fitter weighting a sample is not "
            "equivalent to repeating it (the check's core assertion)"
        ),
        "check_dtype_object": (
            "check body assigns X[0, 0] on the 1-D X requested by the "
            "one_d_array tag"
        ),
        "check_estimator_sparse_array": (
            "the check's data helper converts its 1-D sparse X to lil/dia/bsr, "
            "which scipy does not define for 1-D arrays, and dies before "
            "reaching the estimator (the csr/csc/coo variants and "
            "check_estimator_sparse_matrix/_tag do run, and pass)"
        ),
        "check_fit1d": (
            "1-D x is the native input of this single-feature curve fitter "
            "and is deliberately accepted (promoted to one column), not "
            "rejected with ValueError"
        ),
    }


@parametrize_with_checks(
    [NonlineRegressor()], expected_failed_checks=_expected_failed_checks
)
def test_sklearn_compat(estimator, check):
    check(estimator)
