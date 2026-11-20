"""Unit tests for the unified decoding resolution (Phase 1.6)."""

import pytest

from xai_pilot import config
from xai_pilot.model import _effective_num_beams


@pytest.fixture
def restore_decoding():
    original = config.DECODING
    yield
    config.DECODING = original


def test_decoding_num_beams_mapping():
    assert config.DECODING_NUM_BEAMS["beam"] == 3
    assert config.DECODING_NUM_BEAMS["greedy"] == 1


def test_explicit_num_beams_wins_over_config(restore_decoding):
    config.DECODING = "greedy"
    # An explicit value (e.g. stability/attribution passing num_beams=1) is
    # always honored regardless of the global decoding policy.
    assert _effective_num_beams(3) == 3
    assert _effective_num_beams(1) == 1


def test_none_resolves_from_config_beam(restore_decoding):
    config.DECODING = "beam"
    assert _effective_num_beams(None) == 3


def test_none_resolves_from_config_greedy(restore_decoding):
    config.DECODING = "greedy"
    assert _effective_num_beams(None) == 1
