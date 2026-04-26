from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class HistogramInfo:
    total: int
    mean_brightness: int
    median_brightness: int
    max_brightness: int


@dataclass
class BrightenFactors:
    gain: float
    gamma: float
    low_x: float
    mid_x: float


@dataclass
class BrightnessDetails:
    median_brightness: int
    max_brightness: int
    mean_brightness: int = 0


@dataclass
class LuminanceInfo:
    min_value: int
    median_value: int
    hi_value: int
    noisy: bool = False


@dataclass
class AvgData:
    """State passed between processAvg / updateAvg calls."""
    allocation_out:       Optional[object]     = None
    pixels_rgbf_out:      Optional[np.ndarray] = None
    bitmap_avg_align:     Optional[object]     = None
    allocation_avg_align: Optional[object]     = None
    bitmap_orig:          Optional[object]     = None
    allocation_orig:      Optional[object]     = None
