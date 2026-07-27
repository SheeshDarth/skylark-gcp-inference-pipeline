"""ONNX Runtime wrapper and parser for the published dense pose output."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class ModelContractError(ValueError):
    """Raised when the supplied model does not match the published interface."""


@dataclass(frozen=True)
class Candidate:
    center_x: float
    center_y: float
    width: float
    height: float
    keypoint_x: float
    keypoint_y: float
    confidence: float


def load_model_spec(model_path: str | Path, explicit_path: str | Path | None = None) -> dict[str, Any] | None:
    candidate = Path(explicit_path) if explicit_path else Path(model_path).with_name("model_spec.json")
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"could not read model specification: {candidate}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError("model specification root must be an object")
    return payload


def parse_output_tensor(output: Any) -> list[Candidate]:
    """Parse output0 [1, 9, N] or its unambiguous transposed equivalent.

    The published contract is [batch, channels, candidates] with nine channels:
    xywh, three class probabilities, and keypoint xy. The transposed layout is
    accepted only when the nine-channel axis is unambiguous.
    """

    tensor = np.asarray(output)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        if tensor.shape[1] == 9:
            rows = tensor[0].T
        elif tensor.shape[2] == 9:
            rows = tensor[0]
        else:
            raise ModelContractError(f"expected nine output channels, got shape {tensor.shape}")
    elif tensor.ndim == 2 and 9 in tensor.shape:
        rows = tensor if tensor.shape[1] == 9 else tensor.T
    else:
        raise ModelContractError(f"expected output shape [1,9,N], got {tensor.shape}")

    if rows.shape[1] != 9:
        raise ModelContractError(f"parsed output does not have nine fields: {rows.shape}")

    rows = np.asarray(rows, dtype=np.float32)
    finite = np.all(np.isfinite(rows), axis=1)
    rows = rows[finite]
    if rows.size == 0:
        return []

    scores = np.max(rows[:, 4:7], axis=1)
    keypoints = rows[:, 7:9]
    boxes = rows[:, 0:4]
    candidates: list[Candidate] = []
    for box, point, score in zip(boxes, keypoints, scores, strict=True):
        candidates.append(
            Candidate(
                center_x=float(box[0]),
                center_y=float(box[1]),
                width=float(box[2]),
                height=float(box[3]),
                keypoint_x=float(point[0]),
                keypoint_y=float(point[1]),
                confidence=float(np.clip(score, 0.0, 1.0)),
            )
        )
    return candidates


class OnnxRunner:
    def __init__(self, model_path: str | Path, model_spec_path: str | Path | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised in the container
            raise RuntimeError("onnxruntime is required for inference") from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelContractError(f"model does not exist: {self.model_path}")
        self.spec = load_model_spec(self.model_path, model_spec_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ModelContractError(f"expected one model input, got {len(inputs)}")
        self.input = inputs[0]
        shape = tuple(self.input.shape)
        if (
            len(shape) != 4
            or shape[0] not in (1, "1")
            or shape[1] not in (3, "3")
            or shape[2] not in (640, "640")
            or shape[3] not in (640, "640")
        ):
            raise ModelContractError(f"expected input [1,3,640,640], got name={self.input.name!r} shape={shape}")
        self.input_name = self.input.name
        output_names = [output.name for output in self.session.get_outputs()]
        if "output0" in output_names:
            self.output_index = output_names.index("output0")
        elif len(output_names) == 1:
            self.output_index = 0
        else:
            raise ModelContractError(f"could not identify output0 among {output_names}")
        self.output_names = output_names

    def describe(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "input_shape": list(self.input.shape),
            "input_type": self.input.type,
            "output_names": self.output_names,
            "model_spec_loaded": self.spec is not None,
        }

    def predict(self, tensor: np.ndarray) -> list[Candidate]:
        array = np.asarray(tensor, dtype=np.float32)
        if array.shape != (1, 3, 640, 640):
            raise ModelContractError(f"expected tensor shape [1,3,640,640], got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ModelContractError("input tensor contains non-finite values")
        if np.min(array) < -1e-5 or np.max(array) > 1.00001:
            raise ModelContractError("input tensor must be in [0,1]")
        outputs = self.session.run(None, {self.input_name: array})
        if self.output_index >= len(outputs):
            raise ModelContractError("selected output index is not present in runtime result")
        return parse_output_tensor(outputs[self.output_index])
