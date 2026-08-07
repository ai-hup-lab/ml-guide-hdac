"""Shared building blocks for the HDAC classification pipeline.

Entry points in ``src/`` stay thin and delegate here, so each convention is
written down exactly once:

    io       delimiter sniffing, the two feature-cache layouts, row-count checks
    models   locating heads, the Mordred scaling split, the label decision rule
    metrics  the five reported metrics and the fold-summary statistics
    features molecular fingerprint generators (Mordred/PaDEL optional)
"""
from . import io, metrics, models

__all__ = ['io', 'metrics', 'models']
