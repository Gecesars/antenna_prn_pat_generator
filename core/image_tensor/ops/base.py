from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Type

from ..backend import Backend
from ..tensor_image import TensorImage


_OP_REGISTRY: Dict[str, Type["ImageOp"]] = {}


class ImageOp:
    op_name = "ImageOp"

    def validate(self, img: TensorImage) -> None:
        _ = img

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        raise NotImplementedError

    def params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if str(k).startswith("_"):
                continue
            if callable(v):
                continue
            out[str(k)] = v
        return out

    def signature(self) -> Dict[str, Any]:
        return {"op": str(self.op_name), **self.params()}

    def hash_key(self) -> str:
        payload = json.dumps(self.signature(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return self.signature()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageOp":
        if not isinstance(data, dict):
            raise ValueError("Invalid op payload.")
        op = str(data.get("op", "")).strip()
        if not op:
            raise ValueError("Missing op name.")
        token = op.lower()
        if token not in _OP_REGISTRY:
            raise ValueError(f"Unknown op: {op}")
        impl = _OP_REGISTRY[token]
        kwargs = {k: v for k, v in data.items() if k != "op"}
        return impl(**kwargs)


def register_op(op_cls: Type[ImageOp]) -> Type[ImageOp]:
    name = str(getattr(op_cls, "op_name", op_cls.__name__)).strip().lower()
    _OP_REGISTRY[name] = op_cls
    _OP_REGISTRY[str(op_cls.__name__).strip().lower()] = op_cls
    return op_cls


def get_op_registry() -> Dict[str, Type[ImageOp]]:
    return dict(_OP_REGISTRY)
