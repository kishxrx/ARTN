"""
Tests for ARTN package setup and configuration.
"""

import pytest


def test_package_imports():
    """Verify the ARTN package imports successfully."""
    import src
    
    assert hasattr(src, "__version__")
    assert hasattr(src, "__author__")


def test_version_format():
    """Verify version follows semantic versioning format."""
    import re
    
    import src
    
    # Should match major.minor.patch format
    pattern = r"^\d+\.\d+\.\d+.*$"
    assert re.match(pattern, src.__version__), f"Version {src.__version__} does not match semver pattern"


def test_submodule_imports():
    """Verify all submodules are importable."""
    from src import tokenizer, model, training, inference, evaluation
    
    # All should be modules, not None
    assert tokenizer is not None
    assert model is not None
    assert training is not None
    assert inference is not None
    assert evaluation is not None


@pytest.mark.parametrize(
    "module_name,expected_attrs",
    [
        ("src", ["__version__", "__author__"]),
        ("src.tokenizer", ["__all__"]),
        ("src.model", ["__all__"]),
        ("src.training", ["__all__"]),
        ("src.inference", ["__all__"]),
        ("src.evaluation", ["__all__"]),
    ],
)
def test_module_attributes(module_name: str, expected_attrs: list):
    """Verify modules have expected attributes."""
    import importlib
    
    module = importlib.import_module(module_name)
    for attr in expected_attrs:
        assert hasattr(module, attr), f"{module_name} missing attribute {attr}"
