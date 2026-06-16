"""Logging behaviour of echo_if and the dtfit logger."""

import logging

from dtfit.helpers import FittingOptions, echo_if
from dtfit.log import enable_logging, logger


def test_echo_logs_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="dtfit"):
        echo_if(FittingOptions(echo=True), "visible-message", [1, 2])
    assert "visible-message" in caplog.text
    assert "1" in caplog.text


def test_non_echo_is_silent_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="dtfit"):
        echo_if(FittingOptions(echo=False), "hidden-message")
    assert "hidden-message" not in caplog.text


def test_non_echo_visible_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="dtfit"):
        echo_if(FittingOptions(echo=False), "debug-message")
    assert "debug-message" in caplog.text


def test_enable_logging_returns_dtfit_logger():
    assert enable_logging() is logger
