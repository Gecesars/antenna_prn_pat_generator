from __future__ import annotations

import os
from typing import Dict, Optional

from PIL.PngImagePlugin import PngInfo

from ..tensor_image import TensorImage


def export_png(img: TensorImage, path: str, dpi: int = 300, metadata: Optional[Dict[str, str]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pim = img.to_pil()
    info = PngInfo()
    for k, v in dict(metadata or {}).items():
        info.add_text(str(k), str(v))
    d = int(max(1, dpi))
    pim.save(path, format="PNG", dpi=(d, d), pnginfo=info)
    return path


def export_pdf_asset(img: TensorImage, path: str, dpi: int = 300, metadata: Optional[Dict[str, str]] = None) -> str:
    return export_png(img=img, path=path, dpi=dpi, metadata=metadata)
