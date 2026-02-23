"""
Unit tests — anonymizer interface & classical backends (Task 8.3).

Tests:
  • Anonymizer registry / factory (get_anonymizer, list_anonymizers)
  • GaussianBlurAnonymizer — output shape, dtype, range; deterministic
  • PixelateAnonymizer     — output shape; block_size override
  • BlackoutAnonymizer     — output shape; default is all-black
  • Batch processing via FaceCropBatch → AnonymizedBatch
  • _make_result helper clamps to [0,1]
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.contracts import (
    AnonymizedBatch,
    AnonymizedFace,
    FaceCrop,
    FaceCropBatch,
    FaceCropMeta,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_face(seed: int = 0) -> FaceCrop:
    rng = np.random.RandomState(seed)
    return FaceCrop(
        image=rng.rand(256, 256, 3).astype(np.float32),
        meta=FaceCropMeta(dataset="test", split="test", image_id=f"img_{seed}"),
    )


# ══════════════════════════════════════════════════════════════════════
#  Registry
# ══════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_list_anonymizers(self):
        from src.anonymizers import list_anonymizers
        names = list_anonymizers()
        assert isinstance(names, list)
        assert len(names) >= 3
        assert "blur" in names
        assert "pixelate" in names
        assert "blackout" in names

    def test_get_anonymizer_blur(self):
        from src.anonymizers import get_anonymizer
        anon = get_anonymizer("blur", kernel_size=15)
        assert anon.name == "blur"
        assert anon.configurable_params["kernel_size"] == 15

    def test_get_anonymizer_pixelate(self):
        from src.anonymizers import get_anonymizer
        anon = get_anonymizer("pixelate", block_size=8)
        assert anon.name == "pixelate"

    def test_get_anonymizer_blackout(self):
        from src.anonymizers import get_anonymizer
        anon = get_anonymizer("blackout")
        assert anon.name == "blackout"

    def test_unknown_raises(self):
        from src.anonymizers import get_anonymizer
        with pytest.raises(ValueError, match="Unknown"):
            get_anonymizer("nonexistent_anonymizer_xyz")


# ══════════════════════════════════════════════════════════════════════
#  GaussianBlurAnonymizer
# ══════════════════════════════════════════════════════════════════════

class TestGaussianBlur:
    @pytest.fixture
    def blur(self):
        from src.anonymizers.classical import GaussianBlurAnonymizer
        return GaussianBlurAnonymizer(kernel_size=31)

    def test_output_shape(self, blur):
        face = _make_face()
        result = blur.anonymize_single(face)
        assert isinstance(result, AnonymizedFace)
        assert result.image.shape == (256, 256, 3)

    def test_output_dtype(self, blur):
        result = blur.anonymize_single(_make_face())
        assert result.image.dtype == np.float32

    def test_output_range(self, blur):
        result = blur.anonymize_single(_make_face())
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_deterministic(self, blur):
        face = _make_face(42)
        r1 = blur.anonymize_single(face)
        r2 = blur.anonymize_single(face)
        np.testing.assert_array_equal(r1.image, r2.image)

    def test_output_differs_from_input(self, blur):
        face = _make_face(0)
        result = blur.anonymize_single(face)
        # Blur should change the image
        assert not np.array_equal(result.image, face.image)

    def test_kernel_size_override(self, blur):
        face = _make_face()
        # Use a small kernel
        r_small = blur.anonymize_single(face, kernel_size=3)
        r_large = blur.anonymize_single(face, kernel_size=99)
        # Larger kernel = more blur → images should differ
        assert not np.array_equal(r_small.image, r_large.image)

    def test_keeps_reference(self, blur):
        face = _make_face()
        result = blur.anonymize_single(face)
        assert result.original is face
        assert result.anonymizer_name == "blur"

    def test_even_kernel_corrected(self):
        from src.anonymizers.classical import GaussianBlurAnonymizer
        anon = GaussianBlurAnonymizer(kernel_size=30)
        assert anon.configurable_params["kernel_size"] == 31  # corrected to odd

    def test_small_kernel_corrected(self):
        from src.anonymizers.classical import GaussianBlurAnonymizer
        anon = GaussianBlurAnonymizer(kernel_size=1)
        assert anon.configurable_params["kernel_size"] >= 3


# ══════════════════════════════════════════════════════════════════════
#  PixelateAnonymizer
# ══════════════════════════════════════════════════════════════════════

class TestPixelate:
    @pytest.fixture
    def pixelate(self):
        from src.anonymizers.classical import PixelateAnonymizer
        return PixelateAnonymizer(block_size=16)

    def test_output_shape(self, pixelate):
        result = pixelate.anonymize_single(_make_face())
        assert result.image.shape == (256, 256, 3)

    def test_output_range(self, pixelate):
        result = pixelate.anonymize_single(_make_face())
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_creates_blocks(self, pixelate):
        """With block_size=16, pixels within each 16×16 block should be identical."""
        face = _make_face(42)
        result = pixelate.anonymize_single(face)
        img = result.image
        # Check the top-left 16×16 block — all pixels should be the same
        block = img[:16, :16, :]
        for r in range(16):
            for c in range(16):
                np.testing.assert_array_equal(block[r, c], block[0, 0])

    def test_deterministic(self, pixelate):
        face = _make_face(0)
        r1 = pixelate.anonymize_single(face)
        r2 = pixelate.anonymize_single(face)
        np.testing.assert_array_equal(r1.image, r2.image)


# ══════════════════════════════════════════════════════════════════════
#  BlackoutAnonymizer
# ══════════════════════════════════════════════════════════════════════

class TestBlackout:
    @pytest.fixture
    def blackout(self):
        from src.anonymizers.classical import BlackoutAnonymizer
        return BlackoutAnonymizer(color=(0, 0, 0))

    def test_output_shape(self, blackout):
        result = blackout.anonymize_single(_make_face())
        assert result.image.shape == (256, 256, 3)

    def test_all_black(self, blackout):
        result = blackout.anonymize_single(_make_face())
        assert result.image.max() == 0.0

    def test_white_fill(self):
        from src.anonymizers.classical import BlackoutAnonymizer
        anon = BlackoutAnonymizer(color=(255, 255, 255))
        result = anon.anonymize_single(_make_face())
        assert result.image.min() == 1.0

    def test_deterministic(self, blackout):
        face = _make_face(0)
        r1 = blackout.anonymize_single(face)
        r2 = blackout.anonymize_single(face)
        np.testing.assert_array_equal(r1.image, r2.image)


# ══════════════════════════════════════════════════════════════════════
#  Batch processing
# ══════════════════════════════════════════════════════════════════════

class TestBatchProcessing:
    def test_batch_via_call(self):
        from src.anonymizers.classical import GaussianBlurAnonymizer
        anon = GaussianBlurAnonymizer(kernel_size=15)
        faces = [_make_face(i) for i in range(5)]
        batch = FaceCropBatch(crops=faces)
        result = anon(batch)
        assert isinstance(result, AnonymizedBatch)
        assert len(result) == 5
        assert result.images.shape == (5, 256, 256, 3)

    def test_single_via_call(self):
        from src.anonymizers.classical import PixelateAnonymizer
        anon = PixelateAnonymizer(block_size=8)
        face = _make_face()
        result = anon(face)
        assert isinstance(result, AnonymizedFace)


# ══════════════════════════════════════════════════════════════════════
#  _make_result clamps to [0, 1]
# ══════════════════════════════════════════════════════════════════════

class TestMakeResult:
    def test_clamps_negative(self):
        from src.anonymizers.classical import GaussianBlurAnonymizer
        anon = GaussianBlurAnonymizer(kernel_size=3)
        face = _make_face()
        # Directly test _make_result with out-of-range image
        bad_img = np.full((256, 256, 3), -0.5, dtype=np.float32)
        result = anon._make_result(image=bad_img, original=face)
        assert result.image.min() >= 0.0

    def test_clamps_high(self):
        from src.anonymizers.classical import GaussianBlurAnonymizer
        anon = GaussianBlurAnonymizer(kernel_size=3)
        face = _make_face()
        bad_img = np.full((256, 256, 3), 2.0, dtype=np.float32)
        result = anon._make_result(image=bad_img, original=face)
        assert result.image.max() <= 1.0
