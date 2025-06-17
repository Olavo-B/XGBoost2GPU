"""
XGBoost2GPU - A library to generate CUDA code from XGBoost models using TreeLUT

This library provides tools to convert XGBoost machine learning models into
optimized CUDA code for GPU inference acceleration.

Author: Olavo Alves Barros Silva
Contact: olavo.barros@ufv.com
Date: 2025-06-11
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Olavo Alves Barros Silva"
__email__ = "olavo.barros@ufv.com"
__license__ = "MIT"

# Main imports with error handling
try:
    from .treePruningHash import TreePruningHash
    _TREE_PRUNING_AVAILABLE = True
except ImportError as e:
    TreePruningHash = None
    _TREE_PRUNING_AVAILABLE = False
    import warnings
    warnings.warn(f"TreePruningHash not available: {e}")

try:
    from .xgboost2gpu import XGBoost2GPU
    _XGBOOST2GPU_AVAILABLE = True
except ImportError as e:
    XGBoost2GPU = None
    _XGBOOST2GPU_AVAILABLE = False
    import warnings
    warnings.warn(f"XGBoost2GPU not available: {e}")

# Define what gets imported with "from xgboost2gpu import *"
__all__ = [
    "XGBoost2GPU",
    "TreePruningHash", 
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Package metadata
package_info = {
    "name": "xgboost2gpu",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "description": "A library to generate CUDA code from XGBoost models using TreeLUT",
}


def get_version():
    """Return the version string."""
    return __version__


def get_package_info():
    """Return package information as a dictionary."""
    return package_info.copy()


def check_dependencies():
    """Check if all dependencies are available."""
    status = {
        "XGBoost2GPU": _XGBOOST2GPU_AVAILABLE,
        "TreePruningHash": _TREE_PRUNING_AVAILABLE,
    }
    return status
