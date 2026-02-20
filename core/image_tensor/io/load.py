from __future__ import annotations

from ..tensor_image import TensorImage


def from_file(path: str, device: str = "auto") -> TensorImage:
    return TensorImage.from_file(path, device=device)


def from_bytes(raw: bytes, device: str = "auto") -> TensorImage:
    return TensorImage.from_bytes(raw, device=device)
