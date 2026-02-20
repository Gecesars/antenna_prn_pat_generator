from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from core.image_tensor import Pipeline, TensorImage
from core.image_tensor.io import export_png
from core.image_tensor.ops import (
    ContrastOp,
    BrightnessOp,
    GammaOp,
    GaussianBlurOp,
    UnsharpMaskOp,
)


class ImagePipelineController:
    def __init__(self, cache_root: str = "cache/image_tensor"):
        self.cache_root = os.path.abspath(str(cache_root))
        os.makedirs(self.cache_root, exist_ok=True)

    def build_profile(self, profile: str) -> Pipeline:
        token = str(profile or "pdf_quality").strip().lower()
        if token == "fast_preview":
            ops = [
                ContrastOp(1.05),
                BrightnessOp(1.02),
            ]
            return Pipeline(ops=ops, name="fast_preview")
        if token == "debug_deterministic":
            ops = [
                GammaOp(1.0),
                ContrastOp(1.0),
                BrightnessOp(1.0),
            ]
            return Pipeline(ops=ops, name="debug_deterministic")
        if token == "beautify_datasheet":
            ops = [
                GaussianBlurOp(0.35),
                UnsharpMaskOp(radius=1.2, amount=130, threshold=1),
                ContrastOp(1.08),
                BrightnessOp(1.01),
            ]
            return Pipeline(ops=ops, name="beautify_datasheet")
        # pdf_quality default
        ops = [
            GaussianBlurOp(0.45),
            UnsharpMaskOp(radius=1.4, amount=145, threshold=1),
            ContrastOp(1.10),
            BrightnessOp(1.02),
        ]
        return Pipeline(ops=ops, name="pdf_quality")

    def process_tensor(
        self,
        image: TensorImage,
        profile: str = "pdf_quality",
        device: str = "auto",
        use_cache: bool = True,
    ) -> TensorImage:
        pipe = self.build_profile(profile)
        return pipe.execute(image, device=device, use_cache=use_cache, cache_dir=self.cache_root)

    def process_numpy(
        self,
        array: np.ndarray,
        profile: str = "pdf_quality",
        device: str = "auto",
        use_cache: bool = True,
    ) -> np.ndarray:
        img = TensorImage.from_numpy(array, device=device)
        out = self.process_tensor(img, profile=profile, device=device, use_cache=use_cache)
        return out.to_numpy(dtype="uint8")

    def process_png(
        self,
        input_path: str,
        output_path: str,
        profile: str = "pdf_quality",
        device: str = "auto",
        use_cache: bool = True,
        dpi: int = 300,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        img = TensorImage.from_file(input_path, device=device)
        out = self.process_tensor(img, profile=profile, device=device, use_cache=use_cache)
        return export_png(out.to("cpu"), output_path, dpi=dpi, metadata=metadata)
