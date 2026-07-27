import numpy as np
import pytest

from solution.model import ModelContractError, parse_output_tensor


def test_parse_published_output_layout():
    output = np.zeros((1, 9, 2), dtype=np.float32)
    output[0, 0:4, 0] = [100, 110, 20, 30]
    output[0, 4:7, 0] = [0.1, 0.8, 0.2]
    output[0, 7:9, 0] = [101, 112]
    output[0, 0:4, 1] = [300, 310, 20, 20]
    output[0, 4:7, 1] = [0.2, 0.1, 0.3]
    output[0, 7:9, 1] = [302, 308]

    candidates = parse_output_tensor(output)

    assert len(candidates) == 2
    assert candidates[0].keypoint_x == pytest.approx(101)
    assert candidates[0].keypoint_y == pytest.approx(112)
    assert candidates[0].confidence == pytest.approx(0.8)


def test_parser_rejects_ambiguous_channel_count():
    with pytest.raises(ModelContractError):
        parse_output_tensor(np.zeros((1, 8, 20), dtype=np.float32))

