"""Preprocessing: detection, alignment, landmarks, face parsing."""

# Lazy imports — these modules require cv2 / torch / insightface etc.

_DETECTOR_NAMES = {
    "FaceBox", "FaceDetector", "RetinaFaceDetector", "MTCNNDetector",
    "align_face", "REFERENCE_LANDMARKS_256",
}
_LANDMARK_NAMES = {"LandmarkExtractor", "landmarks_to_heatmaps"}
_PARSER_NAMES = {
    "FaceParser", "colorize_mask", "overlay_mask",
    "get_identity_region_mask", "get_expression_region_mask", "PARSING_LABELS",
}


def __getattr__(name: str):
    if name in _DETECTOR_NAMES:
        from src.preprocess import detector as _det
        return getattr(_det, name)
    if name in _LANDMARK_NAMES:
        from src.preprocess import landmarks as _lm
        return getattr(_lm, name)
    if name in _PARSER_NAMES:
        from src.preprocess import face_parser as _fp
        return getattr(_fp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_DETECTOR_NAMES | _LANDMARK_NAMES | _PARSER_NAMES)
