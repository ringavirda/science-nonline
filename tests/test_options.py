"""FittingOptions behaviour."""

from dtfit.helpers import FittingOptions


def test_obsolete_dsbi_h_removed():
    options = FittingOptions()
    assert not hasattr(options, "dsbi_h")
    assert "dsbi_h" not in options.as_dict()


def test_update_and_as_dict_roundtrip():
    options = FittingOptions()
    options.update({"echo": True, "lasso_alpha": 0.5})
    assert options.echo is True
    assert options.as_dict()["lasso_alpha"] == 0.5


def test_update_ignores_unknown_keys():
    options = FittingOptions()
    options.update({"does_not_exist": 123})
    assert not hasattr(options, "does_not_exist")
