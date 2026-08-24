"""AdaSS -- Adaptive Sparse Steering.

`import adass` gives you everything the notebooks use. Path helpers live in `adass.paths`
and the main ones (`artifact`, `results_path`, `figure`) are re-exported here.
"""
from . import paths
from .paths import artifact, results_path, figure, ROOT
from .core import *          # noqa: F401,F403  -- the public surface, see core.py

__version__ = "0.4.0"
