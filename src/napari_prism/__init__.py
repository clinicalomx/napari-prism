# __version__ = "0.1"
from importlib.metadata import version

__version__ = version("napari-prism")

from . import datasets, gr, im, io, pl, pp, tl

__all__ = ["im", "pp", "tl", "im", "pl", "io", "datasets", "gr"]
