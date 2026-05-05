"""
Tests for foreground_attention package.

Run with: pytest foreground_attention/tests/

These tests cover the standalone (no-torch) parts of the package. The
attention bias function is tested separately when torch is available.
"""

import numpy as np
import pytest
from PIL import Image

from foreground_attention import ForegroundMaskGenerator
from foreground_attention.mask_generator import ForegroundMaskGenerator as MG


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_image():
    """100x100 image with a clear foreground blob (centered dark square on light)."""
    arr = np.full((100, 100, 3), 220, dtype=np.uint8)
    arr[25:75, 25:75] = 30  # dark square in middle
    return Image.fromarray(arr)


@pytest.fixture
def small_image():
    """20x20 image — below default min_side_for_decomp."""
    arr = np.full((20, 20, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def uniform_image():
    """100x100 image with uniform color — OTSU should produce degenerate output."""
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def panoramic_image():
    """300x40 image (ratio 7.5) — typical crosswalk shape."""
    arr = np.full((40, 300, 3), 200, dtype=np.uint8)
    arr[15:25, :] = 50  # dark stripe
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestMaskShape:

    def test_output_shape(self, normal_image):
        gen = MG()
        masks = gen(normal_image, cls='trafficsignal')
        assert masks.shape == (3, 100, 100)
        assert masks.dtype == np.float32

    def test_value_range(self, normal_image):
        gen = MG()
        masks = gen(normal_image, cls='trafficsignal')
        assert masks.min() >= 0.0
        assert masks.max() <= 1.0

    def test_num_regions_configurable(self, normal_image):
        gen = MG(num_regions=5)
        masks = gen(normal_image, cls='trafficsignal')
        assert masks.shape == (5, 100, 100)


class TestFallbacks:

    def test_crosswalk_fallback(self, normal_image):
        gen = MG(fallback_classes=('crosswalk',))
        masks = gen(normal_image, cls='Crosswalk')  # case-insensitive
        # all 3 masks should be identical (all-ones)
        assert (masks[0] == masks[1]).all()
        assert (masks[1] == masks[2]).all()
        assert (masks[0] == 1.0).all()

    def test_resolution_fallback(self, small_image):
        gen = MG(min_side_for_decomp=24)
        masks = gen(small_image, cls='trafficsignal')
        assert (masks[0] == masks[1]).all()
        assert (masks[1] == masks[2]).all()
        assert (masks[0] == 1.0).all()

    def test_uniform_image_fallback(self, uniform_image):
        """Uniform-color images can't be OTSU-thresholded usefully."""
        gen = MG()
        masks = gen(uniform_image, cls='trafficsignal')
        # The mask shape is correct
        assert masks.shape == (3, 100, 100)
        # And all 3 should be the same (degenerate-OTSU fallback to whole-image)
        assert np.allclose(masks[0], masks[1])
        assert np.allclose(masks[1], masks[2])

    def test_class_case_insensitive(self, normal_image):
        gen = MG(fallback_classes=('crosswalk',))
        m_lower = gen(normal_image, cls='crosswalk')
        m_upper = gen(normal_image, cls='CROSSWALK')
        m_mixed = gen(normal_image, cls='CrossWalk')
        assert np.array_equal(m_lower, m_upper)
        assert np.array_equal(m_lower, m_mixed)


class TestNonFallbackBehavior:

    def test_three_strips_for_normal_image(self, normal_image):
        """Normal image should get 3 horizontally-stacked strips, not all-ones."""
        gen = MG()
        masks = gen(normal_image, cls='trafficsignal')
        # not all the same
        assert not np.allclose(masks[0], masks[1])

    def test_strips_are_disjoint_when_blur_is_zero(self, normal_image):
        """With no boundary blur, strips should be strictly disjoint where bg=0."""
        gen = MG(boundary_blur_sigma=0.0)
        masks = gen(normal_image, cls='trafficsignal')
        overlap = np.minimum(masks[0], masks[1]).sum()
        # Some overlap from rounding boundaries; should be << foreground area
        assert overlap < masks[0].sum() * 0.2

    def test_strips_cover_foreground_approximately(self, normal_image):
        """Union of strips should approximate the foreground region."""
        gen = MG(boundary_blur_sigma=0.0)
        masks = gen(normal_image, cls='trafficsignal')
        union = np.maximum.reduce(masks)
        # Foreground is ~25% of image (50x50 in 100x100)
        # Strips inside bbox * intersected with fg should also be ~25%
        assert 0.15 < union.mean() < 0.50


class TestNumpyInput:

    def test_accepts_numpy_array(self):
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        arr[30:70, 30:70] = 50
        gen = MG()
        masks = gen(arr, cls='trafficsignal')
        assert masks.shape == (3, 100, 100)
