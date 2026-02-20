# tensor.md — Módulo Tensor (CPU/GPU) para Imagens e Edições + Suite de Testes Robusta (PySide6)
Versão: 1.0 • Alvo: EFTX Diagram Suite • GPU alvo: RTX 3060 6GB (CUDA) • UI: PySide6

> **Objetivo:** adicionar um backend de processamento de imagens baseado em tensores (CPU/GPU), com pipelines reprodutíveis para exportação (diagramas/tabelas/PDF) e suporte a edição (overlays/markers/máscaras), mantendo fallback completo para CPU e garantindo **qualidade + performance + determinismo** via uma **suite de testes robusta**.

---

## 0) Regras obrigatórias
1) **Não quebrar nada existente.** O módulo tensor deve ser um backend **novo** consumido por rotas/fluxos atuais via adapter.
2) **Fallback automático.** Se CUDA não existir (ou falhar), executar em CPU com mesma API.
3) **Sem “eval” inseguro.** Expressões (markers matemáticos) devem usar parser seguro (whitelist).
4) **Tudo testável.** Toda operação relevante precisa de testes unitários + testes de integração + testes de regressão visual (“golden”).
5) **Não travar UI.** Processamento pesado deve rodar fora do thread do Qt (worker), com cancelamento e progress.

---

## 1) O que este módulo deve entregar
### 1.1 Funcionalidades mínimas (MVP)
- Representação `TensorImage` (H×W×C) com:
  - `device`: `cpu` ou `cuda`
  - `dtype`: `uint8` e `float32` (suporte a `float16` em CUDA quando aplicável)
  - `color`: RGB/RGBA (alpha)
- Operações de imagem (“ops”) com API uniforme (CPU/GPU):
  - Resize (nearest/bilinear/bicubic/lanczos)
  - Crop/Pad
  - Rotate/Affine/Perspective
  - Gamma/Levels/Contrast/Brightness
  - Gaussian blur / Median blur / Bilateral (mínimo: gaussian + median)
  - Unsharp mask (sharpen controlado)
  - Alpha blend (composição overlay)
  - Colormap (opcional MVP)
- Overlays técnicos:
  - Marker point/line/crosshair
  - Text label (com layout consistente)
  - Box de métricas (painel)
- Pipeline reprodutível:
  - Um “graph” linear de operações: `Pipeline([op1, op2, ...])`
  - Serialização/deserialização: JSON (para reproduzir o mesmo resultado)
- Export:
  - PNG com DPI e metadados
  - Assets prontos para PDF (imagens em DPI-alvo: ex. 300/600)

### 1.2 Funcionalidades avançadas (fase 2)
- Batch processing (N imagens) com:
  - cache por hash (imagem + pipeline + versão)
  - execução paralela e/ou stream (processa e descarta intermediários)
- Plugins IA (opcionais):
  - Super-resolution 2× (tiling para caber em 6GB)
  - Denoise DL (opcional)
- “Auto Beautify” para tabelas exportadas:
  - anti-alias por supersampling/downsample
  - reforço de linhas (edge enhance)
  - padronização de fundo/contraste
  - limpeza de ruído leve

---

## 2) Stack recomendada (Windows)
### 2.1 Backend principal (recomendado)
- **PyTorch** (tensores + CUDA) — recomendado por maturidade e ecossistema
- **Pillow** (leitura/gravação básica) e/ou **OpenCV** (opcional)
- **NumPy** (interoperabilidade)

Instalação (dev):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow opencv-python pytest pytest-xdist pytest-benchmark
```

> Observação: a versão CUDA/cuXX deve ser alinhada com o driver NVIDIA do ambiente do usuário. O módulo deve detectar e registrar `torch.version.cuda`, `torch.cuda.get_device_name(0)` e `driver` quando possível.

### 2.2 Alternativa (se não quiser PyTorch)
- CuPy + OpenCV CUDA (mais “clássico”), porém reduz ecossistema de modelos.

**Decisão**: implementar com PyTorch como backend **primário**, mas com design que permita backend alternativo futuro.

---

## 3) Estrutura de pastas proposta (obrigatória)
Criar um pacote isolado e plugável:

```
core/
  image_tensor/
    __init__.py
    backend.py              # seleção CPU/GPU, dtype policy, env checks
    tensor_image.py         # TensorImage + conversões
    ops/
      __init__.py
      base.py               # ImageOp, validação, assinatura, hashing
      geometry.py           # resize/crop/pad/rotate/affine/perspective
      color.py              # gamma/levels/contrast/brightness
      filters.py            # gaussian/median/bilateral/unsharp
      blend.py              # alpha blend, overlay stack
      annotate.py           # markers, text, metric boxes (render)
    pipeline/
      __init__.py
      pipeline.py           # Pipeline([...]) + execute + serialize
      cache.py              # cache por hash e versionamento
    safe_math/
      __init__.py
      parser.py             # parser seguro p/ expressões (whitelist)
      functions.py          # sqrt/log10/sin/cos/...
    io/
      __init__.py
      load.py               # from_file, from_bytes
      save.py               # export png, export assets
    workers/
      __init__.py
      qt_worker.py          # integração com PySide6 (QThreadPool/QRunnable)
tests/
  tensor/
    test_backend_detection.py
    test_tensor_image_roundtrip.py
    test_ops_geometry.py
    test_ops_filters.py
    test_ops_blend_annotate.py
    test_pipeline_serialization.py
    test_safe_math_parser.py
    test_cache_hashing.py
    test_batch_streaming.py
  golden/
    inputs/                 # imagens de entrada (pequenas e controladas)
    expected/               # outputs esperados (png)
  perf/
    bench_resize.py
    bench_pipeline_export.py
```

---

## 4) API pública (contrato obrigatório)
### 4.1 `TensorImage`
Classe (pública) com:
- `TensorImage.from_file(path, device="auto")`
- `TensorImage.from_numpy(np_img, device="auto")`
- `img.to(device)`
- `img.to_numpy()` (CPU)
- `img.clone()`
- `img.shape`, `img.dtype`, `img.device`
- `img.ensure_rgba()` / `img.ensure_rgb()`
- `img.linearize()` (opcional)

**Regras de dtype:**
- Base default: `uint8` (para reduzir VRAM e overhead).
- Converter para `float32/float16` apenas quando a operação exigir (ex.: filtros/warp com interpolação).

### 4.2 `ImageOp`
Interface para cada operação (pattern obrigatório):
- `op.validate(img)`
- `op.apply(img, backend) -> TensorImage`
- `op.signature() -> dict` (parâmetros serializáveis)
- `op.hash_key()` (para cache)

### 4.3 `Pipeline`
- `Pipeline(ops: list[ImageOp])`
- `pipeline.execute(img, device="auto", use_cache=True) -> TensorImage`
- `pipeline.to_json()` / `Pipeline.from_json()`

### 4.4 `Backend`
- `Backend.detect()` decide CPU/CUDA
- `backend.device`, `backend.supports_fp16`, `backend.max_vram_hint`
- `backend.log_env()` retorna string com informações do runtime (para logs e bug reports)

---

## 5) Detecção GPU + política RTX 3060 6GB (obrigatório)
### 5.1 Detecção
- Se `torch.cuda.is_available()`:
  - device = `cuda:0`
  - registrar `name`, `capability`, `total_memory`
- Caso contrário: device = `cpu`

### 5.2 Política de VRAM (6GB)
- Para imagens grandes (>= 4K) ou pipelines longos:
  - preferir execução “stream” (uma imagem por vez)
  - evitar cópias intermediárias (in-place quando seguro)
  - usar `float16` em CUDA onde não degradar qualidade (com tolerância em testes)
- Para super-resolution/denoise:
  - obrigatoriamente usar **tiling** (ex.: 512–1024 px) + blending de bordas

---

## 6) Integração com a aplicação (sem alterar o existente)
### 6.1 Adapter obrigatório
Criar `adapters/image_pipeline_controller.py` (ou equivalente) que:
- recebe imagens/figuras já geradas pelo sistema atual (diagramas e tabelas)
- aplica `Pipeline` predefinida (ex.: `beautify_datasheet`, `pdf_assets_300dpi`)
- retorna PNGs/bytes para o exportador atual (PDF/relatório)

> **Regra:** o código antigo não deve depender de torch diretamente; ele chama o adapter.

### 6.2 Execução em background (PySide6)
- Para chamadas longas:
  - `QThreadPool + QRunnable` ou `QThread` dedicado
  - emitir signals: `progress`, `log`, `result`, `error`, `canceled`
- Cancelamento:
  - flag atômica + checagens entre ops
  - liberar recursos GPU quando cancelar

---

## 7) Overlays e Markers (técnico e reprodutível)
### 7.1 Especificação de overlays (serializável)
Definir um schema JSON (exemplo):
```json
{
  "markers": [
    {"type":"point","x":123,"y":456,"style":{"r":255,"g":0,"b":0,"a":255}, "label":"P1"}
  ],
  "metric_boxes": [
    {"anchor":"top_right","lines":["Ganho: 7.3 dBd","VSWR: <1.1"], "style":{...}}
  ]
}
```

### 7.2 Render consistente
- Fontes: escolher uma fonte padrão (ex.: Segoe UI no Windows) e fallback.
- Tamanho em pt baseado no DPI do alvo (PDF).
- Todas as operações de overlay devem ser testáveis (golden outputs).

---

## 8) Cache e versionamento (obrigatório)
### 8.1 Hash determinístico
- Hash do input (bytes) + hash da pipeline + versão do módulo
- Armazenar em:
  - `cache/image_tensor/<hash>/output.png`
  - `cache/image_tensor/<hash>/meta.json` (device, dtype, timings, versions)

### 8.2 Invalidar cache quando:
- mudar `core/image_tensor` versão (semver interno)
- mudar parâmetros da pipeline
- mudar algoritmo (ex.: nova implementação de resize)

---

## 9) Suite de Testes Robusta (OBRIGATÓRIA)
### 9.1 Framework e organização
- Usar **pytest**
- Usar `pytest-xdist` (paralelo) e `pytest-benchmark` (performance)
- Rodar em dois modos:
  - **CPU-only** (sempre passa)
  - **CUDA** (quando disponível) com tolerâncias

### 9.2 Tipos de teste (obrigatório)
#### (A) Unit tests (lógica/contratos)
- `TensorImage`:
  - roundtrip `from_numpy -> to_numpy` preserva shape/dtype
  - `ensure_rgba/rgb` funciona
  - `to(device)` funciona e não vaza memória (sanity)
- `Backend.detect()`:
  - CPU fallback correto
  - captura de info CUDA (se existir)
- `safe_math`:
  - aceitar expressões permitidas
  - rejeitar tokens proibidos (`__`, `import`, `lambda`, `open`, etc.)
  - testar funções whitelisted

#### (B) Ops tests (correção numérica)
Para cada op:
- validar parâmetros (erros claros)
- invariantes:
  - resize preserva channels
  - crop/pad produz shape exata
  - blur não muda dtype final se política exigir `uint8` output
- CPU vs CUDA:
  - outputs “próximos” dentro de tolerância:
    - `max_abs_diff <= 2` em uint8 (ajustável por op)
    - ou `MSE <= threshold`
  - Em float: `rtol/atol` apropriados

#### (C) Golden tests (regressão visual)
- Manter um conjunto pequeno de imagens em `tests/golden/inputs`
- Para cada pipeline/overlay relevante, gerar output e comparar com `tests/golden/expected`
- Comparação:
  - pixel-perfect quando determinístico
  - caso de variação mínima (GPU/driver): usar métrica:
    - PSNR >= X (ex.: 45 dB)
    - SSIM >= Y (ex.: 0.995) (se implementarem SSIM simples)
  - Salvar diffs (heatmap) quando falhar para debug

> **Obrigatório:** quando golden falhar, o teste deve salvar `out_actual.png` e `out_diff.png` em uma pasta temporária e apontar o caminho no erro.

#### (D) Pipeline tests (serialização e reprodutibilidade)
- `pipeline.to_json()` e `from_json()` são equivalentes
- Hash da pipeline é estável
- Executar pipeline duas vezes com cache ligado:
  - segunda execução deve ser mais rápida e retornar o mesmo resultado

#### (E) Performance tests (benchmarks)
- Benchmark resize 2048×2048
- Benchmark pipeline “beautify + overlay + export”
- Medir CPU vs CUDA (quando houver)
- Critério mínimo (exemplo, ajustar após medir):
  - CUDA deve ser ≥ 1.5× em pipeline de 10 ops para imagem >= 2K
- Registrar tempos no log para histórico

#### (F) Stress / memory tests (sanity)
- Processar 200 imagens 512×512 em loop:
  - não pode crescer memória indefinidamente
- Em CUDA:
  - checar `torch.cuda.max_memory_allocated()` antes/depois (sanity)
  - liberar cache (`torch.cuda.empty_cache()`) quando apropriado (não exagerar)

### 9.3 Execução de testes (comandos)
- CPU:
```bash
pytest -q
```
- CUDA (quando disponível):
```bash
pytest -q -m cuda
```
- Bench:
```bash
pytest -q tests/perf --benchmark-only
```

### 9.4 Marcação e skip inteligente
- Criar marker pytest:
  - `@pytest.mark.cuda` para testes GPU
- Se `torch.cuda.is_available()` for False, **skip** com mensagem clara.

### 9.5 CI local (mínimo)
Mesmo sem CI externo, o agente deve criar scripts:
- `scripts/test_cpu.ps1`
- `scripts/test_cuda.ps1` (se GPU)
- `scripts/bench.ps1`

---

## 10) Critérios de aceite (o agente só “fecha” quando cumprir)
1) Módulo `core/image_tensor` integrado por adapter sem quebrar rotas/export atuais.
2) Pipeline “beautify export” gera PNGs com melhor legibilidade (tabela/linhas/texto) e resultado reprodutível.
3) CPU-only passa **100%** dos testes.
4) CUDA passa testes marcados (quando GPU presente) dentro de tolerâncias.
5) Golden tests implementados e rodando.
6) Benchmarks básicos implementados.
7) Logs de ambiente (torch/cuda/device) disponíveis para bug report.

---

## 11) Observações finais (padrão profissional)
- **Determinismo**: quando possível, definir seeds e flags determinísticas em torch para testes (sem matar performance em produção).
- **Precisão vs velocidade**: disponibilizar “profiles”:
  - `fast_preview`
  - `pdf_quality`
  - `debug_deterministic`
- **Documentar** em `README_TENSOR.md` (curto) como habilitar GPU e rodar testes.

