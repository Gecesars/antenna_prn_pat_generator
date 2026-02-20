from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .backend import get_backend


def _as_float01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    return np.clip(a, 0.0, 1.0)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a.copy()
    if np.issubdtype(a.dtype, np.floating):
        return np.clip(np.round(a * 255.0), 0.0, 255.0).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim != 3:
        raise ValueError("TensorImage expects HxWxC.")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.shape[2] not in (3, 4):
        raise ValueError("TensorImage supports RGB/RGBA.")
    return np.ascontiguousarray(a)


@dataclass
class TensorImage:
    _arr: np.ndarray
    _device: str = "cpu"
    _meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_file(path: str, device: str = "auto") -> "TensorImage":
        img = Image.open(path)
        mode = "RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB"
        arr = np.asarray(img.convert(mode), dtype=np.uint8)
        out = TensorImage(_arr=_ensure_hwc(arr), _device="cpu", _meta={"path": path})
        return out.to(device)

    @staticmethod
    def from_bytes(raw: bytes, device: str = "auto") -> "TensorImage":
        img = Image.open(io.BytesIO(raw))
        mode = "RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB"
        arr = np.asarray(img.convert(mode), dtype=np.uint8)
        out = TensorImage(_arr=_ensure_hwc(arr), _device="cpu", _meta={})
        return out.to(device)

    @staticmethod
    def from_numpy(np_img: np.ndarray, device: str = "auto") -> "TensorImage":
        arr = _ensure_hwc(np.asarray(np_img))
        if arr.dtype not in (np.uint8, np.float32, np.float16):
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            else:
                arr = arr.astype(np.uint8)
        out = TensorImage(_arr=np.ascontiguousarray(arr), _device="cpu", _meta={})
        return out.to(device)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(x) for x in self._arr.shape)

    @property
    def dtype(self) -> str:
        return str(self._arr.dtype)

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def channels(self) -> int:
        return int(self._arr.shape[2])

    def clone(self) -> "TensorImage":
        return TensorImage(_arr=np.array(self._arr, copy=True), _device=self._device, _meta=dict(self._meta or {}))

    def to(self, device: str = "auto") -> "TensorImage":
        b = get_backend(device)
        out = self.clone()
        out._device = str(b.device)
        return out

    def to_numpy(self, dtype: Optional[str] = None) -> np.ndarray:
        arr = np.array(self._arr, copy=True)
        if dtype is None:
            return arr
        token = str(dtype).strip().lower()
        if token in {"uint8", "u8"}:
            return _to_uint8(arr)
        if token in {"float32", "f32"}:
            return _as_float01(arr).astype(np.float32)
        if token in {"float16", "f16"}:
            return _as_float01(arr).astype(np.float16)
        return arr

    def to_pil(self) -> Image.Image:
        arr_u8 = _to_uint8(self._arr)
        return Image.fromarray(arr_u8)

    @staticmethod
    def from_pil(img: Image.Image, device: str = "auto") -> "TensorImage":
        mode = "RGBA" if img.mode == "RGBA" else "RGB"
        arr = np.asarray(img.convert(mode), dtype=np.uint8)
        return TensorImage.from_numpy(arr, device=device)

    def ensure_rgba(self) -> "TensorImage":
        if self.channels == 4:
            return self.clone()
        arr = self.to_numpy()
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        out = np.concatenate([_to_uint8(arr), alpha], axis=2)
        return TensorImage.from_numpy(out, device=self.device)

    def ensure_rgb(self) -> "TensorImage":
        arr = self.to_numpy()
        if arr.shape[2] == 3:
            return self.clone()
        return TensorImage.from_numpy(arr[:, :, :3], device=self.device)

    def linearize(self, gamma: float = 2.2) -> "TensorImage":
        g = max(1e-6, float(gamma))
        arr = _as_float01(self._arr)
        rgb = arr[:, :, :3]
        rgb_lin = np.power(np.clip(rgb, 0.0, 1.0), g)
        if arr.shape[2] == 4:
            out = np.concatenate([rgb_lin, arr[:, :, 3:4]], axis=2)
        else:
            out = rgb_lin
        return TensorImage.from_numpy(out.astype(np.float32), device=self.device)

    def to_torch(self):
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("Torch is not available.") from e
        arr = self.to_numpy(dtype="float32")
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        b = get_backend(self.device)
        if b.device == "cuda":
            ten = ten.to("cuda:0", non_blocking=True)
        return ten

    @staticmethod
    def from_torch(tensor, device: str = "auto") -> "TensorImage":
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("Torch is not available.") from e
        if tensor is None:
            raise ValueError("Invalid tensor.")
        ten = tensor.detach()
        if ten.ndim != 3:
            raise ValueError("Expected CxHxW tensor.")
        if ten.shape[0] not in (3, 4):
            raise ValueError("Expected 3 or 4 channels.")
        if ten.is_cuda:
            ten = ten.cpu()
        arr = ten.permute(1, 2, 0).contiguous().numpy()
        return TensorImage.from_numpy(arr.astype(np.float32), device=device)

    def input_hash(self) -> str:
        h = hashlib.sha256()
        a = np.ascontiguousarray(self._arr)
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes())
        return h.hexdigest()
