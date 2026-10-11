import hashlib

import numpy as np
import pytest
from drivingbench.sim.visual_finish import NATIVE_SHAPE, finish_frame


@pytest.fixture
def inputs():
    original = np.zeros((*NATIVE_SHAPE, 3), np.uint8)
    generated = np.full_like(original, 255)
    mask = np.zeros(NATIVE_SHAPE, bool)
    mask[100:112, 200:212] = True
    return original, generated, mask


def finish(values, **kwargs):
    return finish_frame(*values, pose=(1.2, -3.4, 45), sim_time_s=12, captured_at=123.5, **kwargs)


def test_inward_feather_does_not_change_one_pixel_outside_exact_mask(inputs):
    original, generated, mask = inputs
    original[~mask] = 61
    output, record = finish(inputs, feather_px=2)
    np.testing.assert_array_equal(output[~mask], original[~mask])
    assert np.all(output[100, 200] == 85)
    assert np.all(output[101, 201] == 170)
    assert np.all(output[102, 202] == 255)
    assert record["outside_mask_identical"]
    assert record["generated_outside_mask_changes_ignored"] == (~mask).sum()
    assert not np.shares_memory(output, original)
    assert np.all(generated == 255)
    assert np.all(original[mask] == 0)


def test_zero_feather_exact_replacement_and_canonical_provenance(inputs):
    original, generated, mask = inputs
    output, record = finish(inputs, feather_px=0)
    np.testing.assert_array_equal(output[mask], generated[mask])
    assert record["captured_at"] == 123.5
    assert record["sim_time_s"] == 12
    assert record["pose"] == [1.2, -3.4, 45]
    assert record["live_integration"] is False
    assert record["original_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
    assert record["generated_sha256"] == hashlib.sha256(generated.tobytes()).hexdigest()
    binary = mask.astype(np.uint8) * 255
    again, same = finish((original, generated, binary), feather_px=0)
    np.testing.assert_array_equal(output, again)
    assert record == same
    assert record["mask_sha256"] == hashlib.sha256(binary.tobytes()).hexdigest()
    assert record["output_sha256"] == hashlib.sha256(output.tobytes()).hexdigest()


def test_empty_mask_and_single_pixel_support_are_preserved(inputs):
    original, generated, mask = inputs
    mask[:] = False
    output, record = finish(inputs)
    np.testing.assert_array_equal(output, original)
    assert record["mask_pixels"] == record["changed_pixels"] == 0
    mask[0, 0] = True
    output, record = finish(inputs, feather_px=8)
    assert np.all(output[0, 0] == 28)
    assert record["changed_pixels"] == 1
    assert np.count_nonzero(np.any(output != original, axis=-1)) == 1


@pytest.mark.parametrize(
    "index,value",
    [
        (0, None),
        (1, None),
        (2, None),
        (0, np.zeros((760, 1343, 3), np.uint8)),
        (1, np.zeros((760, 1344, 4), np.uint8)),
        (1, np.zeros((760, 1344, 3), np.float32)),
        (2, np.zeros((760, 1344), np.float32)),
        (2, np.ones((760, 1344), np.uint8)),
    ],
)
def test_missing_resized_or_ambiguous_inputs_are_rejected(inputs, index, value):
    values = list(inputs)
    values[index] = value
    with pytest.raises(ValueError):
        finish(values)


@pytest.mark.parametrize("feather", [-1, 9, 1.5, True])
def test_feather_bound_is_explicit(inputs, feather):
    with pytest.raises(ValueError):
        finish(inputs, feather_px=feather)


@pytest.mark.parametrize(
    "metadata",
    [
        {"pose": [0, 0]},
        {"pose": [0, np.nan, 0]},
        {"sim_time_s": -1},
        {"captured_at": np.inf},
    ],
)
def test_invalid_capture_provenance_cannot_be_published(inputs, metadata):
    kwargs = dict(pose=(0, 0, 0), sim_time_s=1, captured_at=2)
    kwargs.update(metadata)
    with pytest.raises(ValueError):
        finish_frame(*inputs, **kwargs)


def test_offline_unknown_capture_time_stays_unknown(inputs):
    _, record = finish_frame(*inputs, pose=(0, 0, 0), sim_time_s=4, captured_at=None)
    assert record["captured_at"] is None
    assert record["capture_time_policy"].startswith("unknown")
