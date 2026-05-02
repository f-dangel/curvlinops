"""IO collector for detecting and collecting layer inputs/outputs via FX tracing."""

from curvlinops.computers.io_collector.collector import with_kfac_io, with_param_io
from curvlinops.computers.io_collector.layer_io import LayerIO, LayerIOSnapshot

__all__ = ["LayerIO", "LayerIOSnapshot", "with_kfac_io", "with_param_io"]
