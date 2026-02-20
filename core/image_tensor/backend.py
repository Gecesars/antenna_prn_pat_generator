from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Backend:
    device: str
    torch_device: str
    supports_fp16: bool
    max_vram_hint_mb: int
    torch_available: bool
    cuda_available: bool
    torch_version: str
    cuda_version: str
    gpu_name: str

    @staticmethod
    def detect(prefer_device: str = "auto") -> "Backend":
        req = str(prefer_device or "auto").strip().lower()
        env_req = str(os.getenv("EFTX_IMAGE_DEVICE", "")).strip().lower()
        if env_req:
            req = env_req

        torch_available = False
        cuda_available = False
        torch_version = ""
        cuda_version = ""
        gpu_name = ""
        fp16 = False
        vram_mb = 0

        try:
            import torch  # type: ignore

            torch_available = True
            torch_version = str(getattr(torch, "__version__", "") or "")
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
            if cuda_available:
                try:
                    gpu_name = str(torch.cuda.get_device_name(0) or "")
                except Exception:
                    gpu_name = ""
                try:
                    props = torch.cuda.get_device_properties(0)
                    total = int(getattr(props, "total_memory", 0))
                    vram_mb = max(0, int(total // (1024 * 1024)))
                    major = int(getattr(props, "major", 0))
                    fp16 = bool(major >= 5)
                except Exception:
                    pass
        except Exception:
            torch_available = False

        if req in {"cuda", "gpu"} and cuda_available:
            device = "cuda"
        elif req in {"cpu"}:
            device = "cpu"
        else:
            device = "cuda" if cuda_available else "cpu"

        torch_device = "cuda:0" if device == "cuda" else "cpu"
        return Backend(
            device=device,
            torch_device=torch_device,
            supports_fp16=bool(fp16 and device == "cuda"),
            max_vram_hint_mb=int(vram_mb),
            torch_available=bool(torch_available),
            cuda_available=bool(cuda_available),
            torch_version=torch_version,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
        )

    def log_env(self) -> str:
        parts = [
            f"device={self.device}",
            f"torch_available={self.torch_available}",
            f"cuda_available={self.cuda_available}",
            f"torch={self.torch_version or '-'}",
            f"cuda={self.cuda_version or '-'}",
            f"gpu={self.gpu_name or '-'}",
            f"fp16={self.supports_fp16}",
            f"vram_mb={self.max_vram_hint_mb}",
        ]
        return " | ".join(parts)


_DEFAULT_BACKEND: Optional[Backend] = None


def get_backend(prefer_device: str = "auto") -> Backend:
    global _DEFAULT_BACKEND
    req = str(prefer_device or "auto").strip().lower()
    if req not in {"auto", ""}:
        return Backend.detect(req)
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = Backend.detect("auto")
    return _DEFAULT_BACKEND
