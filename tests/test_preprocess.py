import numpy as np

from solution.preprocess import letterbox_rgb


def test_square_letterbox_is_identity_without_padding():
    rgb = np.full((3, 640, 640), 0.25, dtype=np.float32)

    tensor, meta = letterbox_rgb(rgb)

    assert tensor.shape == (3, 640, 640)
    assert np.allclose(tensor, rgb)
    assert meta.scale == 1.0
    assert (meta.pad_x, meta.pad_y) == (0, 0)


def test_partial_window_uses_centered_padding_and_records_inverse():
    rgb = np.zeros((3, 320, 640), dtype=np.float32)

    tensor, meta = letterbox_rgb(rgb)

    assert tensor.shape == (3, 640, 640)
    assert (meta.pad_x, meta.pad_y) == (0, 160)
    assert np.allclose(tensor[:, :160], 114.0 / 255.0)
    assert np.allclose(tensor[:, 160:480], 0.0)
    assert np.allclose(tensor[:, 480:], 114.0 / 255.0)
    assert (320 * meta.scale, 640 * meta.scale) == (320.0, 640.0)
