# AEDT Live (HFSS) — Integração PyAEDT 2025.2 (R2) — Especificação + Plano de Implementação

## Objetivo

Adicionar **interação ao vivo** com **Ansys Electronics Desktop (AEDT) 2025.2** (foco em **HFSS**) à suíte de diagramas (PAT Converter), **sem refatorar o que já existe**: o que já funciona permanece intacto; a integração entra como um **plugin** (nova aba) que:

1. **Conecta** ao AEDT/HFSS (iniciar ou anexar sessão existente).
2. **Lista** projetos/designs/setups/sweeps relevantes.
3. **Executa pós-processamento** (principal):
   - extrai **cortes 2D** (VRP/HRP) diretamente do HFSS;
   - extrai **padrão 3D** (malha θ×φ do ganho) como grid.
4. **Faz parser** dos dados para:
   - **Library**: salvar como diagrama (VRP/HRP) e como artefato 3D (arquivo `.npz` + registro);
   - **Projeto/Workspace**: carregar o VRP/HRP extraído para o estado atual do app e permitir seguir o pipeline existente (composição/exports/visualizações).

> Implementação entregue como pacote “drop-in” (novo código isolado + 1 hook de registro do plugin).

---

## Pré-requisitos técnicos (hard requirements)

### Software
- AEDT **2025.2** instalado com HFSS.
- PyAEDT compatível com AEDT 2025.2 (na prática, usa o pacote `ansys.aedt.core` / PyAEDT).
  - **Preferência**: executar o Python do **próprio AEDT** (ou o Python configurado pela Ansys), para evitar mismatch de DLL.

### Execução
- Windows 10/11 (alvo principal), GPU RTX 3060 6GB (sem impacto direto no pós, mas ok).
- App GUI rodando com CustomTkinter.

---

## Arquitetura de integração (sem quebrar o existente)

### Estratégia “plugin”
- Criar um pacote novo: `eftx_aedt_live/` (sem tocar em módulos existentes).
- Criar um entrypoint: `aedt_live_plugin.py` com função:
  - `register_aedt_live_tab(app, tabview, tab_name="AEDT Live", output_dir=None)`

**Única mudança no app existente**: **importar e chamar** `register_aedt_live_tab` após criar o seu `CTkTabview`.

---

## Fluxo de usuário (UX)

### Aba “AEDT Live”
- Campos:
  - **AEDT version**: default `2025.2`
  - **Project (.aedt)**: selecionar via browse
  - **Design**: texto (opcional)
  - **Setup sweep**: ex. `Setup1 : LastAdaptive`
  - **Sphere**: ex. `3D_Sphere`
  - **Expr**: default `dB(GainTotal)` (editável)
  - **Freq**: opcional (`0.8GHz`, etc.)
- Botões:
  - `Connect`, `Disconnect`
  - `Refresh` (varrer setups/sweeps do projeto)
  - `Pull VRP` (corte em Theta, Phi=0deg → conversão Theta→Elevação)
  - `Pull HRP` (corte em Phi, Theta=90deg)
  - `Pull 3D` (Theta×Phi grid; exporta `.npz` e `.obj`)

### Comportamento dos botões
- **NUNCA** travar a UI:
  - Todas as ações que falam com AEDT rodam em **thread** (worker) e retornam via `after()`.

---

## Pós-processamento: extração dos dados

### Cortes 2D
**VRP (vertical)**:
- Primário: `Theta`
- Fixos: `Phi=0deg` (ou `90deg`, configurável no futuro)
- Conversão para “elevação”:
  - `elev = 90 - theta`  → eixo final `[-90, +90]`

**HRP (horizontal)**:
- Primário: `Phi`
- Fixos: `Theta=90deg` (plano do horizonte)

### Padrão 3D (grid θ×φ)
- Objetivo: obter um grid `values[theta_index][phi_index]`
- Estratégias robustas:
  1. **Preferida**: `hfss.post.get_solution_data(report_category="Far Fields", context=sphere)` com variações `"Theta":"All"`, `"Phi":"All"`.
  2. **Fallback**: varrer `Phi` e extrair cortes de `Theta`, interpolando em uma grade fixa (mais lento, porém determinístico).

---

## Parser e compatibilidade com a suíte (Library e Projeto)

### Estrutura de dado para 2D (compatível com o app)
```json
{
  "name": "AEDT_VRP_Setup1_LastAdaptive",
  "type": "VRP",
  "angles": [ ... ],
  "values": [ ... ],
  "meta": { ... }
}
```

### Inserção no **workspace/projeto**
- Preferido: `app.load_from_library(pattern_dict)`
- Fallback: setar `app.v_angles/app.v_mags` ou `app.h_angles/app.h_mags` e chamar `app.plot_diagrams()`.

### Inserção na **library**
- Tentar chamar (por reflexão):
  - `app.db_manager.add_diagram(...)`, ou `app.db.add_diagram(...)`, etc.
- Se não existir, salvar ao menos em disco (export).

### Padrão 3D
- Salvar em disco como `.npz`:
  - `theta_deg`, `phi_deg`, `values`, `meta`
- Opcional: exportar `.obj` para visualização externa e compatibilidade com pipeline.

---

## Pacote implementado (drop-in)

Foi gerado um pacote com:
- `eftx_aedt_live/`:
  - `session.py`: conexão HFSS (PyAEDT)
  - `farfield.py`: extração 2D/3D com fallback
  - `export.py`: exporta CSV/JSON/NPZ/OBJ
  - `bridge.py`: envia dados para library/workspace (reflexão)
  - `worker.py`: thread worker para Tk
  - `ui_tab.py`: aba “AEDT Live”
- `aedt_live_plugin.py`: função de registro do plugin (hook único)

---

## Integração no app existente (alteração mínima e localizada)

1) Copiar o pacote para a raiz do repositório:
- `aedt_live_plugin.py`
- pasta `eftx_aedt_live/`

2) No arquivo principal da GUI (onde você cria o `CTkTabview`):
```python
from aedt_live_plugin import register_aedt_live_tab

# depois de criar self.tabs (CTkTabview):
register_aedt_live_tab(app=self, tabview=self.tabs, tab_name="AEDT Live", output_dir=self.output_dir)
```

3) Rodar o app.

---

## Requisitos de robustez (para o agente)

- **Zero freeze**: toda chamada PyAEDT em worker.
- **Timeouts**: adicionar watchdog se necessário (opcional).
- **Logs claros**: tudo que dá erro deve ir para log + popup.
- **Não destruir sessões**: por padrão `close_on_exit=False`.
- **Fallbacks**: se `get_solution_data` falhar → usar `reports_by_category.far_field` → se falhar → erro legível.
- **Caches** (futuro): cache de grids e cortes por chave `(project, design, setup, sphere, expr, freq, phi/theta)`.

---

## Próximas extensões (já previstas)

1) Listar automaticamente:
   - projetos abertos;
   - designs HFSS do projeto;
   - nomes de “Infinite Sphere”/radiation setup.
2) Botão “Analyze” (solve) para setup selecionado.
3) Importação de “markers” diretamente de máximos/zeroes do corte extraído.
4) Export para o formato nativo de diagramas do seu app (PRN/PAT), se necessário.

