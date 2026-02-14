# null_fill.md — Implementação de cálculo de *null fill* para composição vertical de antenas

## 0) Objetivo (o que este módulo faz)

Implementar um **módulo de síntese de excitação** (pesos complexos) para uma **composição vertical** (stack) de antenas, visando **preenchimento de nulos** no diagrama de elevação.

O módulo deve produzir, a partir de:
- **geometria** (posições/espacamentos verticais),
- **frequência**,
- **modelo de elemento** (opcional, mas recomendado),
- **região(s) de nulos** a preencher (ângulos e piso em dB),
- e um modo de controle (**amplitude**, **fase**, ou **ambos**),

os **valores aplicáveis** no *harness*:
- **amplitudes** (fração de potência por baia / atenuação equivalente em dB),
- **fases** (graus),
- e **comprimentos elétricos relativos** (metros) para um cabo com VF informado.

> Premissa: o “valor a aplicar” em cada baia é um **peso complexo**  
> \[
> w_n = a_n e^{j\phi_n}
> \]
> que se traduz em: potência ∝ \(a_n^2\) e ajuste de fase ∝ \(\phi_n\).

---

## 1) Definições e convenções (obrigatórias para consistência)

### 1.1. Sistema de coordenadas e ângulos

- Arranjo ao longo do **eixo vertical** \(z\).
- Elevação \(\varepsilon\) medida a partir do horizonte:
  - \(\varepsilon = 0^\circ\) → horizonte
  - \(\varepsilon = 90^\circ\) → zênite

### 1.2. Constantes

- \(c_0 = 299\,792\,458\ \text{m/s}\)
- \(\lambda_0 = c_0 / f\)
- \(k = 2\pi / \lambda_0\)

### 1.3. Fator de arranjo (array factor)

Para \(N\) baias (ou elementos), com posições \(z_n\) e pesos \(w_n\):

\[
AF(\varepsilon)=\sum_{n=0}^{N-1} w_n \cdot e^{j k z_n \sin\varepsilon}
\]

Diagrama total (aproximação separável):

\[
E(\varepsilon) \approx E_{\text{elem}}(\varepsilon)\cdot AF(\varepsilon)
\]

- Se **não houver modelo do elemento**, usar \(E_{\text{elem}}(\varepsilon)=1\).
- Se houver, multiplique no objetivo (alvo) ou na avaliação final.

---

## 2) Entradas do usuário (contrato de interface)

### 2.1. Geometria
- `z_m`: lista/array em metros com as alturas \(z_n\) (tamanho N), **ou**
- `spacing_m` + `N`: caso uniforme, gerar \(z_n = n \cdot spacing\).

### 2.2. Frequência
- `f_hz`: frequência em Hz.

### 2.3. Especificação de *null fill*
O usuário deve informar **uma ou mais regiões**:
- `fill_bands`: lista de bandas, cada uma com:
  - `eps_min_deg`, `eps_max_deg`
  - `floor_db` (piso mínimo permitido; ex.: −14 dB)
  - opcional: `weight` (importância relativa)

E também:
- `mainlobe_tilt_deg` (opcional): onde você deseja o pico (tilt).  
  *Se omitido, assume que o horizonte (≈ 0–1°) é referência do “máximo”*.

### 2.4. Modo de controle (obrigatório)
- `mode = "amplitude"` → ajustar **somente amplitude** (fases fixas)
- `mode = "phase"` → ajustar **somente fase** (amplitudes fixas)
- `mode = "both"` → ajustar amplitude e fase livre (pesos complexos)

> O sistema deve **obrigar** o usuário a escolher um destes três.

### 2.5. Restrições (recomendadas)
- `amp_limits_db`: limites de atenuação por baia (ex.: [0, 6] dB).
- `phase_limits_deg`: limites de fase relativa (ex.: ±180°).
- `power_norm`: `"sum_abs2_1"` (padrão), ou `"max_1"`.

### 2.6. Conversão para *harness*
- `vf`: velocidade de propagação do cabo (0.66, 0.78, 0.85…)
- `ref_index`: índice da baia referência para fase (padrão: 0)

---

## 3) Saídas (o que o agente deve produzir)

Para cada baia \(n\):

- `w[n]`: peso complexo final (normalizado).
- `a[n] = |w[n]|` (amplitude)
- `phi_deg[n] = angle(w[n])` (fase em graus, referenciada)
- `p_frac[n] = |w[n]|^2 / sum(|w|^2)` (fração de potência)
- `att_db[n] = -10 log10(p_frac[n] / p_ref)` (opcional, se escolher referência)
- `delta_len_m[n]`: comprimento adicional do cabo para realizar `phi_deg[n]`

Além disso:
- `AF(eps)` e `E(eps)` (curvas amostradas)
- Métricas: pior nulo dentro das bandas, pico, sidelobe máx (se avaliado)

---

## 4) Estratégias de síntese a implementar (três níveis)

### 4.1. Nível A (robusto e geral): mínimos quadrados regularizado (pesos complexos)

#### 4.1.1. Discretização do ângulo
Escolher uma grade de elevação:
- `eps_grid_deg`: ex.: 0° a 30° com passo 0.1° (ajustável)

Converter:
\[
\varepsilon_m = \text{deg2rad}(\text{eps\_grid\_deg}[m])
\]

#### 4.1.2. Matriz de “steering”
\[
A_{m,n} = e^{j k z_n \sin(\varepsilon_m)}
\]

Opcional (element pattern):
- Se `E_elem(eps)` existir, pode ser incorporado multiplicando o alvo `d[m]`
  ou ajustando a métrica (recomendado: aplicar no alvo).

#### 4.1.3. Definição do alvo de magnitude (*null fill target*)
Construir `mag_target[m]` (linear) a partir das bandas:
- Default fora das bandas: “curva suave” que preserva mainlobe.
- Dentro de cada banda \([eps_min, eps_max]\): impor
  \[
  |E(\varepsilon)| \ge 10^{\frac{floor\_db}{20}}
  \]
  (note **20** no denominador para magnitude de campo; se trabalhar em potência, usar 10.)

> Regra prática: se o usuário fornece piso em **dB de campo**, use 20; se em **dB de potência**, use 10.  
> **Decisão do produto:** padronize `floor_db` como **dB de campo**.

#### 4.1.4. Alvo complexo (fase do alvo)
Como normalmente o usuário só define magnitude, defina a fase do alvo `d[m]` assim:

- Inicialize `w0`:
  - uniforme: \(w0=[1,1,\dots,1]\), ou
  - progressivo para tilt: \(w0[n]=e^{-jk z_n \sin(\varepsilon_{\text{tilt}})}\)

- Use a fase do padrão inicial como referência:
  \[
  \angle d_m \leftarrow \angle(A w0)_m
  \]
- Então:
  \[
  d_m = \text{mag\_target}_m \cdot e^{j\angle d_m}
  \]

Isso evita “invenção” de fase inconsistente.

#### 4.1.5. Problema de otimização (Tikhonov)
Resolver:

\[
w = \arg\min_w \|A w - d\|^2 + \lambda \|w\|^2
\]

Solução fechada:

\[
(A^H A + \lambda I)w = A^H d
\]

Implementar com:
- `np.linalg.solve`
- \(\lambda\) pequeno (ex.: 1e-3 a 1e-2) com opção de ajuste.

#### 4.1.6. Impor modo de controle (amplitude / fase / ambos)
Após obter `w` (complexo):

- **mode="both"**  
  → aceitar `w` como está, apenas normalizar e aplicar limites.

- **mode="amplitude"** (amplitude livre, fase fixa)  
  - Defina uma fase fixa `phi_fixed[n]`:
    - opção 1 (default): progressiva para tilt
      \[
      \phi_{\text{fixed},n} = -k z_n \sin(\varepsilon_{\text{tilt}})
      \]
    - opção 2: todas iguais (0°)
  - Ajuste amplitude:
    \[
    a_n \leftarrow |w_n|
    \quad,\quad
    w_n \leftarrow a_n e^{j\phi_{\text{fixed},n}}
    \]

- **mode="phase"** (fase livre, amplitude fixa)  
  - Defina amplitude fixa `a_fixed[n]`:
    - default: uniforme (1)
    - ou taper predefinido (Dolph-Chebyshev, Taylor, etc. — opcional)
  - Ajuste fase:
    \[
    \phi_n \leftarrow \angle w_n
    \quad,\quad
    w_n \leftarrow a_{\text{fixed},n} e^{j\phi_n}
    \]

> Observação importante: se você “trava” fase ou amplitude, o resultado pode violar o piso.  
> Portanto, após impor o modo, rode uma **iteração curta**:
> - re-sintetize novamente mantendo a restrição (ver 4.1.7).

#### 4.1.7. Iteração projetada (opcional, recomendado)
Para lidar com `mode="amplitude"` ou `"phase"`:

1) resolva `w` livre (both)  
2) projete para o subespaço permitido (amplitude-only ou phase-only)  
3) atualize a fase do alvo com `angle(A w)` e resolva de novo  
4) repita 3–10 iterações, ou até convergir.

Isso é simples e melhora muito o resultado.

---

### 4.2. Nível B (industrial): sub-arrays (dois grupos) com 2 graus de liberdade

Caso o sistema físico só permita controle por grupos (ex.: 2 divisores principais):

Dividir em dois sub-arranjos:
- grupo 1: índices `G1`
- grupo 2: índices `G2`

\[
AF(\varepsilon)=AF_1(\varepsilon)+c\cdot AF_2(\varepsilon)
\]
onde \(c=\rho e^{j\psi}\) é amplitude+fase relativa.

Para um ângulo dominante \(\varepsilon_0\) (onde existe o nulo mais crítico):
- calcule \(S_0=AF_1(\varepsilon_0)\)
- calcule \(T_0=AF_2(\varepsilon_0)\)
- imponha um alvo \(AF_{\text{des}}(\varepsilon_0)\) com magnitude piso:
\[
c=\frac{AF_{\text{des}}(\varepsilon_0)-S_0}{T_0}
\]

O módulo deve:
- permitir `method="subarray_2"` quando `N` é grande e o usuário só tem um ajuste.

Saídas:
- razão de potência entre grupos: \(|c|^2\)
- fase relativa \(\angle c\)
- e os pesos por baia dentro de cada grupo (normalmente uniformes ou com taper fixo).

---

### 4.3. Nível C (tapers clássicos): Taylor/Chebyshev + pequeno *phase offset*
Implementar como alternativa “rápida”:
- escolher taper de amplitude (Taylor/Chebyshev) para reduzir sidelobes
- aplicar um pequeno desbalanceamento de fase entre metades (ou entre níveis alternados)
para levantar vales.

Isso não substitui a síntese LSQ, mas dá um “fallback”.

---

## 5) Conversão dos pesos para valores de *harness* (obrigatório)

### 5.1. Normalização
Default:
\[
\sum |w_n|^2 = 1
\]
(garante total de potência normalizada)

### 5.2. Fração de potência
\[
p_n = \frac{|w_n|^2}{\sum |w|^2}
\]

### 5.3. Fase e comprimento elétrico
- fase em radianos: \(\phi_n=\angle w_n\)
- referenciar em uma baia `ref_index`:
  \[
  \Delta\phi_n = (\phi_n - \phi_{\text{ref}})\ \bmod\ 2\pi
  \]

Comprimento guiado:
- \(\lambda_g = VF \cdot \lambda_0\)
- \[
\Delta \ell_n = \frac{\Delta\phi_n}{2\pi}\lambda_g
\]

> Importante: `delta_len_m` é **comprimento adicional** relativo.  
> O projeto físico escolhe um “base length” e soma \(\Delta\ell_n\).

---

## 6) Critérios de validação (o agente deve checar e reportar)

### 6.1. Métrica de piso
Para cada banda:
- amostrar `eps_grid` dentro da banda
- medir:
  \[
  \min_{\varepsilon \in \text{banda}} 20\log_{10}|E(\varepsilon)|
  \]
- comparar com `floor_db` (com tolerância ex.: 0.5 dB)

### 6.2. Pico e tilt
- encontrar pico principal e ângulo do pico
- comparar com `mainlobe_tilt_deg` (se fornecido)

### 6.3. Robustez numérica
- evitar singularidade: usar `reg_lambda`
- checar `cond(AHA)` e alertar se muito alto

---

## 7) API sugerida (assinaturas)

### 7.1. Função principal
```python
def synth_null_fill_vertical(
    f_hz: float,
    z_m: np.ndarray,
    eps_grid_deg: np.ndarray,
    fill_bands: list[dict],
    mode: str,                  # "amplitude" | "phase" | "both"
    mainlobe_tilt_deg: float | None = None,
    elem_pattern: callable | None = None,   # elem_pattern(eps_deg)->complex/float
    reg_lambda: float = 1e-3,
    max_iters: int = 8,
    amp_limits_db: tuple[float, float] | None = None,
    phase_limits_deg: float | None = None,
    norm: str = "sum_abs2_1",
) -> dict:
    ...
```

### 7.2. Conversão para harness
```python
def weights_to_harness(
    w: np.ndarray, f_hz: float, vf: float, ref_index: int = 0
) -> dict:
    ...
```

---

## 8) Fluxo de UI (wizard) — perguntas que o software deve fazer

1) **Frequência** (Hz / MHz)  
2) **N e espaçamento** ou lista `z`  
3) **Modo**: amplitude | fase | ambos  
4) **Tilt desejado** (opcional)  
5) **Região(s) para null fill**:
   - banda: eps_min, eps_max, piso (dB de campo)
6) (Opcional) limites práticos:
   - atenuação máxima por baia
   - limite de fase
   - topologia disponível (por baia ou por grupos)
7) **VF do cabo** (para converter fase→comprimento)

O wizard deve mostrar:
- preview do AF inicial (uniforme)
- preview do AF final
- tabela de potência/fase/comprimento

---

## 9) Implementação (código-base mínimo) — referência para o agente

### 9.1. Síntese LSQ regularizada + projeção por modo
```python
import numpy as np

def _build_A(f_hz, z_m, eps_deg):
    c0 = 299_792_458.0
    lam = c0 / f_hz
    k = 2*np.pi / lam
    eps = np.deg2rad(eps_deg)
    return np.exp(1j * k * z_m[None, :] * np.sin(eps)[:, None]), k, lam

def _initial_w(z_m, k, tilt_deg=None):
    if tilt_deg is None:
        return np.ones(len(z_m), dtype=complex)
    eps0 = np.deg2rad(tilt_deg)
    return np.exp(-1j * k * z_m * np.sin(eps0))

def _mag_target(eps_deg, fill_bands, default_floor_linear=0.0):
    # default: não impõe piso fora das bandas
    mt = np.full_like(eps_deg, fill_value=default_floor_linear, dtype=float)
    for b in fill_bands:
        e0, e1 = b["eps_min_deg"], b["eps_max_deg"]
        floor_db = b["floor_db"]   # dB de campo
        floor_lin = 10**(floor_db/20.0)
        mask = (eps_deg >= e0) & (eps_deg <= e1)
        mt[mask] = np.maximum(mt[mask], floor_lin)
    return mt

def _solve_tikhonov(A, d, reg_lambda):
    AH = A.conj().T
    M = AH @ A + reg_lambda * np.eye(A.shape[1])
    b = AH @ d
    return np.linalg.solve(M, b)

def _project_mode(w, mode, z_m, k, tilt_deg=None, amp_fixed=None):
    if mode == "both":
        return w
    if mode == "amplitude":
        # fase fixa progressiva (tilt) ou 0
        if tilt_deg is None:
            phi_fixed = np.zeros_like(z_m, dtype=float)
        else:
            eps0 = np.deg2rad(tilt_deg)
            phi_fixed = -k * z_m * np.sin(eps0)
        a = np.abs(w)
        return a * np.exp(1j * phi_fixed)
    if mode == "phase":
        if amp_fixed is None:
            amp_fixed = np.ones(len(z_m), dtype=float)
        phi = np.angle(w)
        return amp_fixed * np.exp(1j * phi)
    raise ValueError("mode inválido")

def synth_null_fill_vertical(
    f_hz, z_m, eps_grid_deg, fill_bands, mode,
    mainlobe_tilt_deg=None, reg_lambda=1e-3, max_iters=8
):
    A, k, lam = _build_A(f_hz, z_m, eps_grid_deg)

    w = _initial_w(z_m, k, mainlobe_tilt_deg)
    mag_t = _mag_target(eps_grid_deg, fill_bands, default_floor_linear=0.0)

    for _ in range(max_iters):
        # fase alvo baseada no padrão atual
        phase_ref = np.angle(A @ w)
        # alvo: magnitude mínima (piso) + fase coerente
        d = np.maximum(np.abs(A @ w), mag_t) * np.exp(1j * phase_ref)

        w_free = _solve_tikhonov(A, d, reg_lambda)
        w = _project_mode(w_free, mode, z_m, k, mainlobe_tilt_deg)

        # normalização soma |w|^2 = 1
        w = w / np.sqrt(np.sum(np.abs(w)**2))

    AF = A @ w
    return {"w": w, "AF": AF, "eps_deg": eps_grid_deg, "lambda0": lam}
```

> Nota: o alvo `d` acima usa “no mínimo” o piso, mas preserva o que já está acima (não “achata” todo o padrão).

### 9.2. Conversão para *harness*
```python
def weights_to_harness(w, f_hz, vf, ref_index=0):
    c0 = 299_792_458.0
    lam0 = c0 / f_hz
    lamg = vf * lam0

    a = np.abs(w)
    phi = np.angle(w)
    phi_ref = phi[ref_index]

    dphi = (phi - phi_ref) % (2*np.pi)
    delta_len = (dphi/(2*np.pi)) * lamg

    p_frac = (a**2) / np.sum(a**2)

    return {
        "amp": a,
        "phase_deg": np.rad2deg(dphi),
        "p_frac": p_frac,
        "delta_len_m": delta_len,
        "lambda0_m": lam0,
        "lambda_g_m": lamg,
    }
```

---

## 10) Testes mínimos (o agente deve implementar)

### 10.1. Caso base: 4 baias, espaçamento 0.8λ
- `N=4`
- `spacing = 0.8*lambda`
- banda de preenchimento: 2°–8° com piso −14 dB
- comparar `mode="amplitude"`, `"phase"`, `"both"`

Critérios:
- `both` deve atingir o piso mais facilmente
- `amplitude` geralmente reduz nulo, mas pode subir sidelobes
- `phase` pode ser suficiente se o nulo é “de interferência” por simetria

### 10.2. Regressão: consistência de normalização
- sempre `sum(|w|^2)=1` dentro de tolerância 1e-6.

### 10.3. Robustez
- variar `reg_lambda` e garantir que não explode (NaN/Inf).

---

## 11) Observações práticas (para o usuário final — deve ser exibido na UI)

- **Null fill é trade-off:** preenche vales, mas pode aumentar lobos secundários.
- Erros de fase pequenos podem destruir o resultado → medir o harness com VNA.
- Se o sistema físico não permite pesos por baia, use **sub-arrays** (método 4.2).

---

## 12) Checklist de entrega

O agente deve entregar:
1) Módulo Python com as funções acima (com docstrings e type hints).
2) Rotina de plot (matplotlib) para:
   - AF inicial vs AF final (em dB)
   - marcação das bandas de null fill
3) Exportação em JSON/CSV da tabela:
   - baia, p_frac, phase_deg, delta_len_m
4) Testes automatizados (pytest) para os casos do item 10.

---

## 13) “Pergunta obrigatória” ao usuário (para evitar configuração errada)

Antes de rodar o cálculo, a UI deve perguntar:

> Você deseja ajustar **amplitude**, **fase**, ou **amplitude+fase**?

E deve explicar em uma linha:
- Amplitude: altera potência por baia (divisores/atenuadores).
- Fase: altera comprimento elétrico (linhas/cabos).
- Ambos: melhor performance, porém maior complexidade de harness.
