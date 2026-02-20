from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

# Ensure op registry is populated.
from .. import ops as _ops  # noqa: F401
from ..backend import get_backend
from ..ops.base import ImageOp
from ..tensor_image import TensorImage
from .cache import MODULE_VERSION, PipelineCache


ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCheck = Optional[Callable[[], bool]]


def _coerce_op(item: Any) -> ImageOp:
    if isinstance(item, ImageOp):
        return item
    if isinstance(item, dict):
        return ImageOp.from_dict(item)
    raise ValueError(f"Invalid op: {type(item).__name__}")


class Pipeline:
    def __init__(self, ops: Sequence[Any], name: str = "pipeline"):
        self.name = str(name or "pipeline")
        self.ops: List[ImageOp] = [_coerce_op(x) for x in list(ops)]

    def signature(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ops": [op.signature() for op in self.ops],
        }

    def to_json(self) -> str:
        return json.dumps(self.signature(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(payload: str) -> "Pipeline":
        data = json.loads(str(payload or "{}"))
        if not isinstance(data, dict):
            raise ValueError("Invalid pipeline payload")
        ops = data.get("ops", [])
        if not isinstance(ops, list):
            raise ValueError("Invalid pipeline ops list")
        name = str(data.get("name", "pipeline") or "pipeline")
        return Pipeline(ops=ops, name=name)

    def execute(
        self,
        img: TensorImage,
        device: str = "auto",
        use_cache: bool = True,
        cache_dir: str = "cache/image_tensor",
        progress_cb: ProgressCallback = None,
        cancel_check: CancelCheck = None,
    ) -> TensorImage:
        backend = get_backend(device)
        source = img.to(backend.device)
        cache = PipelineCache(cache_dir)
        key = cache.key_for(source.input_hash(), self.signature(), MODULE_VERSION)

        if use_cache:
            hit = cache.get(key)
            if hit is not None:
                out_img, _meta = hit
                if progress_cb:
                    progress_cb(len(self.ops), len(self.ops), "cache_hit")
                return out_img.to(backend.device)

        t0 = time.perf_counter()
        cur = source
        total = max(1, len(self.ops))
        for idx, op in enumerate(self.ops, start=1):
            if callable(cancel_check) and bool(cancel_check()):
                raise RuntimeError("Pipeline cancelled.")
            op.validate(cur)
            cur = op.apply(cur, backend).to(backend.device)
            if progress_cb:
                progress_cb(idx, total, str(op.op_name))

        dt = max(0.0, time.perf_counter() - t0)
        if use_cache:
            meta = {
                "name": self.name,
                "version": MODULE_VERSION,
                "device": backend.device,
                "timing_s": dt,
                "ops": [op.signature() for op in self.ops],
                "dpi": 300,
            }
            cache.put(key, cur.to("cpu"), meta)
        return cur

    def execute_from_numpy(
        self,
        array,
        device: str = "auto",
        use_cache: bool = True,
        cache_dir: str = "cache/image_tensor",
        progress_cb: ProgressCallback = None,
        cancel_check: CancelCheck = None,
    ):
        img = TensorImage.from_numpy(array, device=device)
        return self.execute(
            img=img,
            device=device,
            use_cache=use_cache,
            cache_dir=cache_dir,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
        )
