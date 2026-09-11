import pytest


def pytest_runtest_setup(item):
    """Skip tests marked with gpu or fipy if the required packages are not available."""
    if 'gpu' in item.keywords:
        pytest.importorskip('cupy')
    if 'fipy' in item.keywords:
        pytest.importorskip('fipy')
