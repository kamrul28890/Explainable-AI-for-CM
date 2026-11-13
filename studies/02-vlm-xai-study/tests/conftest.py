"""Shared fixtures and the gpu marker.

GPU tests are skipped unless --gpu is passed, so the suite stays green on the
development machine (see docs/setup/SETUP.md, "Which machine runs what").
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False,
                     help="run tests that require a CUDA GPU and model weights")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA and downloaded weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="needs --gpu (CUDA machine only)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
