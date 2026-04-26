from enum import Enum


class TonemappingAlgorithm(Enum):
    CLAMP       = "CLAMP"
    EXPONENTIAL = "EXPONENTIAL"
    REINHARD    = "REINHARD"
    FU2         = "FU2"
    ACES        = "ACES"


class HistogramType(Enum):
    TYPE_RGB       = "RGB"
    TYPE_LUMINANCE = "LUMINANCE"
    TYPE_VALUE     = "VALUE"
    TYPE_INTENSITY = "INTENSITY"
    TYPE_LIGHTNESS = "LIGHTNESS"


class HDRAlgorithm(Enum):
    """High-level algorithm selector passed to HDRProcessor."""
    STANDARD  = "STANDARD"
    MERTENS   = "MERTENS"
    AVERAGING = "AVERAGING"
