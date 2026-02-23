"""Data contracts and dataset adapters."""

from src.data.contracts import (
    CANONICAL_RESOLUTION,
    CANONICAL_COLOR_SPACE,
    ExpressionLabel,
    EXPRESSION_NAMES,
    FaceCropMeta,
    FaceCrop,
    FaceCropBatch,
    AnonymizedFace,
    AnonymizedBatch,
    uint8_to_float,
    float_to_uint8,
    gray_to_rgb,
)

# Lazy imports — dataset adapters require cv2/pandas which may not be installed yet


def __getattr__(name: str):
    if name == "FER2013Dataset":
        from src.data.fer2013_adapter import FER2013Dataset
        return FER2013Dataset
    if name == "PinsFaceDataset":
        from src.data.pins_adapter import PinsFaceDataset
        return PinsFaceDataset
    if name == "CelebAHQDataset":
        from src.data.celebahq_adapter import CelebAHQDataset
        return CelebAHQDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CANONICAL_RESOLUTION",
    "CANONICAL_COLOR_SPACE",
    "ExpressionLabel",
    "EXPRESSION_NAMES",
    "FaceCropMeta",
    "FaceCrop",
    "FaceCropBatch",
    "AnonymizedFace",
    "AnonymizedBatch",
    "uint8_to_float",
    "float_to_uint8",
    "gray_to_rgb",
    "FER2013Dataset",
    "PinsFaceDataset",
    "CelebAHQDataset",
]
