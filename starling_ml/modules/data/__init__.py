"""Источники данных и адаптеры DataLoader."""

from .loader import DataLoaderSource
from .synthetic import (
    SyntheticSegmentationBatch,
    SyntheticSegmentationData,
    SyntheticSegmentationSpec,
)

__all__ = [
    "DataLoaderSource",
    "SyntheticSegmentationBatch",
    "SyntheticSegmentationData",
    "SyntheticSegmentationSpec",
]
