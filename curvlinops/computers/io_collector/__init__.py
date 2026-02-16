"""IO collector for detecting and collecting layer inputs/outputs via FX tracing."""

from curvlinops.computers.io_collector.collector import with_kfac_io, with_param_io

__all__ = ["with_param_io", "with_kfac_io"]
