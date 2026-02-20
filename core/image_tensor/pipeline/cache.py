from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..io.save import export_png
from ..tensor_image import TensorImage


MODULE_VERSION = "1.0.0"


def _json_hash(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheItem:
    key: str
    image_path: str
    meta_path: str


class PipelineCache:
    def __init__(self, root: str = "cache/image_tensor"):
        self.root = os.path.abspath(str(root))
        os.makedirs(self.root, exist_ok=True)

    def key_for(self, input_hash: str, pipeline_signature: Dict[str, Any], module_version: str = MODULE_VERSION) -> str:
        blob = {
            "input": str(input_hash),
            "pipeline": pipeline_signature,
            "version": str(module_version),
        }
        return _json_hash(blob)

    def paths(self, key: str) -> CacheItem:
        folder = os.path.join(self.root, key)
        return CacheItem(
            key=key,
            image_path=os.path.join(folder, "output.png"),
            meta_path=os.path.join(folder, "meta.json"),
        )

    def get(self, key: str) -> Optional[Tuple[TensorImage, Dict[str, Any]]]:
        p = self.paths(key)
        if not (os.path.isfile(p.image_path) and os.path.isfile(p.meta_path)):
            return None
        try:
            with open(p.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            img = TensorImage.from_file(p.image_path, device="cpu")
            return img, meta if isinstance(meta, dict) else {}
        except Exception:
            return None

    def put(self, key: str, image: TensorImage, meta: Dict[str, Any]) -> CacheItem:
        p = self.paths(key)
        os.makedirs(os.path.dirname(p.image_path), exist_ok=True)
        export_png(image, p.image_path, dpi=int(meta.get("dpi", 300)))
        with open(p.meta_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return p
