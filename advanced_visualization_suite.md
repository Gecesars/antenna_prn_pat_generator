# Advanced Visualization Upgrade — Suite de Diagramas EFTX (2D + 3D + Markers + Math)

Este documento instrui um agente a **adicionar visualização avançada** aos diagramas (inspirado no HFSS):
- **2D**: polar/planar com **markers m1..mN**, tabela de markers, tabela de deltas (Δθ, Δmag), beamwidth XdB, pk2pk, avg/max.
- **3D**: reconstrução e visualização **interativa** do diagrama 3D **a partir dos arquivos 2D** (VRP/HRP) com modos de reconstrução (Omni / Separable / Harmonic-fit opcional).
- **Math functions personalizáveis** aplicadas a markers e/ou ao padrão (expressões seguras + funções builtin).
- **Arquitetura organizada**: core (dados + reconstrução + métricas) separado da UI (interações + render).

> Nota física importante: **um padrão 3D não é unicamente determinável** a partir de apenas 2 cortes (VRP e HRP).  
> A implementação deve oferecer **modos explícitos** (assunções) e indicar isso na UI.

---

## 1) Definição de Pronto (DoD)

### 1.1. Visualização 2D “HFSS-like”
- [ ] Click esquerdo adiciona marker (m1, m2, …), com snap ao ponto mais próximo do vetor.
- [ ] Arrastar marker move e atualiza tabelas.
- [ ] Dupla seleção A/B mostra **Δθ, Δmag** e funções derivadas.
- [ ] Painel/overlay com:
  - Tabela de markers: `Name | Theta(deg) | Ang(deg) | Mag (dB/lin)`
  - Tabela de deltas: `d(m3,m4) | Δθ | ΔAng | ΔMag`
- [ ] Beamwidth XdB (ex.: 3 dB, 10 dB) calculado por “corte” e exibido em tabela.

### 1.2. Visualização 3D
- [ ] Botão “3D View” abre viewer 3D interativo (PyVista ou Plotly).
- [ ] O padrão 3D é gerado a partir de VRP/HRP por modos:
  - Omni (VRP somente)
  - Separable (VRP×HRP)
  - (Opcional) Harmonic-fit (regularizado)
- [ ] Superfície com:
  - **raio proporcional à magnitude** (linear ou dB mapeado)
  - **cor em dB** com colorbar (min/max configuráveis)
  - Overlays: cortes φ=0° e φ=90° (linhas/planos)
- [ ] Click no 3D cria marker (θ,φ,mag) e sincroniza com o painel 2D.

### 1.3. Math engine personalizável
- [ ] Usuário cria “funções” via expressão segura (sem eval livre), salva em JSON.
- [ ] Funções podem usar:
  - marker(s): A, B, etc.
  - constantes e parâmetros (ex.: XdB=10)
  - utilitários: wrap, deg↔rad, log10, abs, min/max, etc.
- [ ] Resultados aparecem em uma tabela “Derived”.

### 1.4. Robustez / Qualidade
- [ ] Sem crash em clique fora do eixo, sem dados, ou padrão vazio.
- [ ] Interpolação com limites e wrap coerentes.
- [ ] Testes unitários para:
  - conversões de ângulo
  - beamwidth XdB
  - reconstrução 3D (Omni e Separable)
  - math engine (parser + exec seguro)

---

## 2) Modelo de dados canônico

### 2.1. PatternCut (já recomendado)
Representa um corte 2D:

```python
@dataclass(frozen=True)
class PatternCut:
    type: Literal["H", "V"]          # H: azimute, V: elevação
    angles_deg: np.ndarray           # H: [-180,180], V: [-90,90] (interno)
    values_lin: np.ndarray           # magnitude linear normalizada (E/Emax)
    meta: dict                       # freq, gain, source, etc.
```

### 2.2. SphericalPattern (novo)
Representa um padrão 3D em grade regular (θ, φ):

```python
@dataclass(frozen=True)
class SphericalPattern:
    theta_deg: np.ndarray            # 0..180 (polar: 0 zênite, 180 nadir)
    phi_deg: np.ndarray              # -180..180 ou 0..360 (definir 1 padrão)
    mag_lin: np.ndarray              # shape (Nt, Np), linear normalizado
    meta: dict
```

**Convenção obrigatória:**
- θ (theta) é o ângulo polar HFSS (0 no +Z, 90 no horizonte, 180 no −Z)
- φ (phi) é o azimute (0 no +X; range interno preferido: **[-180, 180]**)

---

## 3) Conversões e mapeamentos (um único lugar: `core/angles.py`)

### 3.1. Elevação (ε) ↔ Theta (θ)
Se seu VRP interno usa elevação `ε ∈ [-90, 90]`, com 0° no horizonte:

- θ = 90° − ε
- ε = 90° − θ

Validar:
- ε=+90 (zênite) -> θ=0
- ε=0 (horizonte) -> θ=90
- ε=-90 (nadir) -> θ=180

### 3.2. Wrap de φ (azimute)
- `wrap_phi_deg(x)` retorna em [-180,180]:
  ```python
  return ((x + 180) % 360) - 180
  ```

### 3.3. Distância angular mínima (para Δθ em azimute)
- `ang_dist_deg(a,b)`:
  ```python
  d = abs(((a - b + 180) % 360) - 180)
  ```

---

## 4) Reconstrução 3D a partir de VRP/HRP (core)

Criar `core/reconstruct3d.py` com:

```python
def reconstruct_spherical(
    cut_v: PatternCut | None,
    cut_h: PatternCut | None,
    mode: str,                      # "omni" | "separable" | "harmonic"
    theta_deg: np.ndarray,          # ex.: np.linspace(0,180,361)
    phi_deg: np.ndarray,            # ex.: np.linspace(-180,180,361)
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> SphericalPattern:
    ...
```

### 4.1. Modo A — Omni (VRP apenas)
Assume omnidirecional no azimute:

- Converta `cut_v(ε)` para `Vθ(θ)`:
  - para cada θ, ε = 90 − θ
  - interpolate `cut_v.values_lin(ε)`
- Depois:
  - mag_lin[θ, φ] = Vθ(θ)

### 4.2. Modo B — Separable (VRP × HRP)
Assume separabilidade (aproximação prática):

- Obtenha `Vθ(θ)` do VRP
- Obtenha `Hφ(φ)` do HRP
- Combine (duas opções; implementar ambas):
  1) **Produto direto normalizado**:
     \[
     E(\theta,\phi)=\frac{V(\theta)^\alpha\;H(\phi)^\beta}{\max(H^\beta)}
     \]
  2) **Produto com preservação de pico**:
     normalize V e H para max=1, depois:
     \[
     E(\theta,\phi)=\left(V(\theta)^\alpha\cdot H(\phi)^\beta\right)^{1/2}
     \]
- Normalize global para max=1 no final.

**UI deve permitir** escolher opção (1/2) e α, β.

### 4.3. Modo C — Harmonic-fit (opcional, fase 2)
Fit em esféricos com regularização (L2) usando constraints dos cortes.
- Não é obrigatório na primeira entrega; mantenha como stub com TODO.

---

## 5) Renderização 3D (UI)

Há duas alternativas. Implementar A agora e deixar B como export.

### 5.1. A — PyVista (recomendado, robusto)
- Abre janela 3D externa (não embutir em Tk na primeira versão).
- Cria mesh em coordenadas cartesianas:

Para cada (θ, φ):
- converter para rad
- definir raio `r = r0 + scale * f(mag)`
  - `f(mag)` pode ser `mag_lin**gamma` (default gamma=1)
  - ou mapear dB para [0,1] com floor e max

Cartesiano:
- x = r * sinθ * cosφ
- y = r * sinθ * sinφ
- z = r * cosθ

Cor:
- `mag_db = 20*log10(mag_lin+eps)` (ou 10*log10 se potência)
- set scalar = mag_db para colormap + colorbar

Requisitos do viewer:
- Colorbar com Min/Max configuráveis
- Wireframe opcional
- Overlay de cortes φ=0 e φ=90 (linhas)
- Suporte a “pick” (click) -> marker

### 5.2. B — Plotly (fallback/export)
- Gerar HTML com `plotly.graph_objects.Surface` ou `Mesh3d`.
- Abrir em browser ou em webview (opcional depois).
- Exportar HTML junto do projeto.

---

## 6) Markers: modelo, UI, e sincronização 2D/3D

### 6.1. Dataclass Marker
```python
@dataclass
class Marker:
    name: str               # "m1", "m2", ...
    kind: str               # "2D" ou "3D"
    cut: str | None         # "HRP" | "VRP" quando 2D
    theta_deg: float | None # para 3D ou VRP convertido
    phi_deg: float | None   # para 3D ou HRP
    ang_deg: float          # ang no contexto do plot (x-axis/polar theta)
    mag_lin: float
    mag_db: float
```

### 6.2. Regras de criação
- 2D HRP:
  - marker guarda `phi_deg` (ang_deg) e mag.
- 2D VRP:
  - marker guarda `eps_deg` (ang_deg) e mag; opcionalmente compute theta=90-eps.
- 3D:
  - marker guarda θ e φ.

### 6.3. Snap e interpolação
- Ao clicar, buscar o ponto mais próximo no array **ou** interpolar.
- Para precisão “HFSS-like”, faça:
  - snap por padrão (ponto discreto)
  - opção: “interpolated marker” (checkbox)

### 6.4. Tabelas (como na imagem)
Criar painel “Markers” com:
- tabela markers (m1..mN)
- tabela deltas:
  - d(m_i, m_j) com:
    - Δang (wrap se HRP)
    - Δmag_dB
- botões:
  - Clear, Rename, Delete
  - Copy table
  - Export CSV

---

## 7) Math Functions personalizáveis (engine seguro)

Criar `core/math_engine.py` com:
- Parser baseado em AST (permitir apenas nós seguros):
  - BinOp, UnaryOp, Call, Name, Constant, Attribute
  - bloquear: import, subscripts perigosos, comprehensions, lambdas, etc.
- Funções permitidas: `abs, min, max, sqrt, log10, sin, cos, tan, atan2`
- Constantes: `pi, e`
- Helpers: `wrap_phi, ang_dist`

### 7.1. Variáveis disponíveis na expressão
- `A`, `B` (markers selecionados)
  - `A.ang_deg`, `A.mag_db`, `A.mag_lin`, etc.
- `params` dict (ex.: `params["xdb"]=10`)
- `pattern` (opcional):
  - expor apenas métodos seguros (ex.: `pattern.sample(theta,phi)`)

### 7.2. Exemplos de expressões (ship as presets)
1) Δθ (HRP wrap):
   - `ang_dist(A.ang_deg, B.ang_deg)`
2) ΔMag:
   - `B.mag_db - A.mag_db`
3) Beamwidth XdB (em torno do pico A):
   - `beamwidth_xdb(cut, peak_ang=A.ang_deg, xdb=params["xdb"])`
4) Midpoint:
   - `wrap_phi((A.ang_deg + B.ang_deg)/2)`

> Implementar `beamwidth_xdb()` em `core/metrics.py` com interpolação linear.

### 7.3. Persistência
- salvar funções do usuário em:
  - `%APPDATA%/EFTX/PATConverter/marker_math.json`
- cada função:
  - `name`, `expr`, `params_schema`, `applies_to` ("HRP"/"VRP"/"3D")

---

## 8) Métricas avançadas “HFSS-like” (tabela max/avg/pk2pk/beamwidth)

Criar `core/metrics_advanced.py`:
- `max_db`, `avg_db` (média em dB não é física; oferecer também avg em linear)
- `pk2pk_db = max_db - min_db`
- `beamwidth_xdb(angles, mag_db, xdb, around_peak=True)`
- `first_nulls`, `sll` (sidelobe level) opcional

UI: caixa/tabela no canto (como HFSS) com:
- série/cut atual (ex.: “dB(GainTotal), Phi=0°”)
- max, avg, pk2pk, BW_10dB

---

## 9) UI/UX: onde colocar isso na aplicação

### 9.1. Aba nova: “Visualização Avançada”
Adicionar uma aba com layout 2 colunas:
- Esquerda: 2D plot (VRP ou HRP selecionável) + overlays (markers + rings)
- Direita: painéis:
  - Markers
  - Deltas
  - Derived (math)
  - Reconstruction 3D (mode, params) + botão “Open 3D Viewer”
Rodapé: status bar (cursor readout)

### 9.2. Integração com abas existentes
- “Arquivo”: botão `Advanced View` que abre aba com o cut atual.
- “Composição Vertical/Horizontal”: após gerar composto, disponibiliza para advanced view e 3D.

---

## 10) Implementação organizada (arquivos)

Criar/ajustar módulos:

```
core/
  angles.py
  reconstruct3d.py
  metrics.py                # beamwidth_xdb aqui
  metrics_advanced.py
  math_engine.py
ui/
  tabs/tab_advanced_viz.py
  widgets/plot_panel.py     # já previsto
  widgets/marker_table.py
  widgets/derived_table.py
  viewers/viewer3d_pyvista.py
  interactions/plot_interactor.py
```

---

## 11) Picking (click) no 3D para marker

### 11.1. PyVista pick
- Use `plotter.enable_point_picking(callback=...)`
- Callback recebe ponto (x,y,z). Converter de volta para θ, φ:
  - r = sqrt(x^2+y^2+z^2)
  - θ = arccos(z/r)
  - φ = atan2(y,x)
- Sample mag no grid mais próximo e criar marker.

Sincronizar:
- Atualizar tabela markers
- Se estiver com corte HRP/VRP ativo, também mostrar projeção no 2D (opcional)

---

## 12) Exportações relacionadas ao 3D
Adicionar “Export 3D”:
- `.vtp` (VTK PolyData) (PyVista/VTK)
- `.obj` (malha)
- `.html` (Plotly)
- snapshot `.png` (viewer)

---

## 13) Testes (mínimo)
- `test_angles.py`: theta↔eps, wrap, ang_dist
- `test_beamwidth_xdb.py`: casos sintéticos
- `test_reconstruct3d_omni.py`: 3D omni gera mag constante em φ
- `test_reconstruct3d_separable.py`: confere outer product + normalização
- `test_math_engine.py`: bloqueia AST perigoso, permite expressões válidas

---

## 14) Observações de engenharia (erros típicos a evitar)
1) **dB de campo vs potência**: defina o padrão (recomendado: valores_lin = campo).  
   - dB = 20·log10(mag_lin + eps)
2) **wrap de φ**: todas as comparações e deltas em HRP devem usar distância circular.
3) **theta vs elevação**: não misturar VRP ε com θ do 3D.
4) **floor**: limite inferior em dB (ex.: -50 dB) para evitar “buracos” e NaN.
5) **performance**: precompute grid e use numpy; não loop em Python para 361×361.

---

## 15) Roadmap (fase 2)
- Harmonic-fit com regularização (reconstrução mais realista)
- Importar padrões 3D reais (HFSS export 3D, CST, etc.)
- Embedding do 3D dentro da janela (Qt/WebView) se desejado
- Comparador multi-padrão (overlay de 2–4 séries)

---

## Entrega
O agente deve:
1) Implementar `reconstruct3d.py` (Omni + Separable) e testes.
2) Implementar “Advanced Visualization” tab com markers e métricas avançadas.
3) Implementar viewer 3D (PyVista) com pick → marker.
4) Implementar math engine com presets e persistência.
5) Garantir que tudo funciona com os arquivos 2D já suportados (VRP/HRP/PAT/PRN/CSV/TXT).
