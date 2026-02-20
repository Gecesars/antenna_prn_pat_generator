# Image Tensor Backend

This project includes a pluggable image processing backend in `core/image_tensor`.

## Features
- `TensorImage` abstraction (H x W x C, RGB/RGBA)
- Backend detection with CPU/CUDA fallback
- Serializable operation pipeline
- Cache by input hash + pipeline signature + module version
- Safe math parser (AST whitelist)
- Qt worker for non-blocking execution
- Adapter for app integration: `adapters/image_pipeline_controller.py`

## Runtime behavior
- If CUDA is available (PyTorch + GPU), backend selects `cuda` in `auto` mode.
- If CUDA is unavailable or fails, backend automatically uses CPU.
- Existing export/report flows continue working even if tensor post-process fails.

## Profiles
- `fast_preview`
- `pdf_quality`
- `beautify_datasheet`
- `debug_deterministic`

## Tests
CPU:
```powershell
./scripts/test_cpu.ps1
```

CUDA (when available):
```powershell
./scripts/test_cuda.ps1
```

Benchmarks:
```powershell
./scripts/bench.ps1
```

## Optional GPU install
Example (Windows CUDA 12.1 wheels):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
