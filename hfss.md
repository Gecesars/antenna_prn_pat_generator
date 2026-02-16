# hfss.md — Integração ao vivo com ANSYS AEDT/HFSS 2025.2 via PyAEDT (instalar, testar e executar)

Este documento instrui o **agente** a **instalar**, **integrar**, **testar de verdade** e **operar** a nova funcionalidade **AEDT Live (HFSS)** no app, com foco em **pós-processamento** (diagramas 2D e 3D de far-field), **sem alterar o que já existe** — apenas adicionando uma aba nova e “bridges” de import para **Library** e **Projeto (workspace)**.

> Arquivos já fornecidos nesta conversa (devem ser usados pelo agente):
- `aedt_live.md` — especificação/DoD/arquitetura do módulo AEDT Live.
- `eftx_aedt_live_plugin.zip` — implementação “drop-in” do plugin AEDT Live.

---

## 0) Definições e metas (DoD de integração)

### DoD mínimo (aceitação)
1) App exibe a nova aba **AEDT Live** (sem quebrar nenhuma aba existente).
2) Conecta ao **AEDT 2025.2**:
   - **Attach** (anexa ao AEDT aberto) e/ou
   - **New Session** (abre nova sessão).
3) Lista corretamente:
   - Projetos (`.aedt`), designs HFSS, setups, e nome(s) da Infinite Sphere.
4) Extrai e importa para o app:
   - **HRP** (Phi sweep, Theta=90°)
   - **VRP** (Theta sweep, Phi fixo 0° ou 90°)
   - **3D θ×φ** (grid) do far-field do modelo selecionado
5) “Send to Library” grava o dataset na **biblioteca SQLite** do app (ou mecanismo equivalente).
6) “Send to Project” injeta o dataset no **workspace/projeto atual** do app (VRP/HRP/3D) e atualiza plots.
7) O app **não congela** durante operações AEDT (tudo em worker thread).
8) Evidências de teste real:
   - artefatos exportados (`hrp.json`, `vrp.json`, `ff_3d.npz`/`json`)
   - logs (timing e sucesso/falha por etapa)
   - screenshots (aba AEDT Live com HRP/VRP plotados)

---

## 1) Pré-requisitos (Windows recomendado)

### 1.1. Software
- **ANSYS Electronics Desktop (AEDT) 2025 R2 / 2025.2** instalado, incluindo **HFSS**.
- Licença ativa (local ou license server).

### 1.2. Python (2 opções suportadas)
**Opção A — Python do AEDT (recomendado)**  
Maior estabilidade COM/gRPC e libs do Ansys.

**Opção B — Python do seu venv do app**  
Funciona, mas pode exigir ajustes (PATH/COM).

> O agente deve tentar **A primeiro** para validar a integração. Depois, testar no venv do app.

### 1.3. Biblioteca
- PyAEDT (via `ansys.aedt.core`).
  - Instalação típica: `pip install pyaedt`

---

## 2) Instalação: dependências e validação do ambiente

### 2.1. Criar branch
```bash
git checkout -b feat/aedt-live-hfss-2025_2
```

### 2.2. Instalar PyAEDT no ambiente correto
#### Se usar o Python do AEDT:
1) Abrir o console python do AEDT (Ansys Python/Automation).
2) Rodar:
```bash
python -m pip install --upgrade pip
python -m pip install pyaedt
```

#### Se usar venv do app:
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pyaedt
```

### 2.3. Smoke test de “subir desktop”
Rodar no mesmo Python que pretende usar no app:

```bash
python -c "import ansys.aedt.core as a; d=a.Desktop(specified_version='2025.2', non_graphical=True, new_desktop_session=True, close_on_exit=False); print('AEDT OK'); d.release_desktop(); print('released')"
```

**PASS:** imprime `AEDT OK` e `released`.  
**FAIL:** pare e corrija ambiente/licença.

---

## 3) Instalar o plugin no repositório (sem alterar core)

### 3.1. Extrair o zip do plugin para `plugins/aedt_live/`
Estrutura alvo:

```
plugins/
  aedt_live/
    aedt_live_plugin/
      __init__.py
      ...
```

Passos:
1) Criar a pasta:
```bash
mkdir -p plugins/aedt_live
```
2) Extrair `eftx_aedt_live_plugin.zip` para `plugins/aedt_live/`.

> O agente deve confirmar que existe `register_aedt_live_tab(...)` no módulo.

### 3.2. Garantir import do plugin
Se o app não roda como package, adicione `plugins` ao `sys.path` no entrypoint (mínimo, sem refatorar):

```python
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plugins"))
```

---

## 4) Integrar a aba “AEDT Live” (1 hook, sem mexer no resto)

### 4.1. Localizar onde o CTkTabview é criado
No arquivo principal do app, achar o ponto em que `self.tabs = CTkTabview(...)` é montado e as abas são adicionadas.

### 4.2. Inserir o hook do plugin
Adicionar apenas:

```python
from plugins.aedt_live.aedt_live_plugin import register_aedt_live_tab

register_aedt_live_tab(
    app=self,
    tabview=self.tabs,
    tab_name="AEDT Live",
    output_dir=getattr(self, "output_dir", None),
)
```

**Regras:**
- Não alterar abas existentes.
- Se `output_dir` não existir, criar fallback somente quando `None`.

---

## 5) Bridges obrigatórios: importar para Library e Projeto

O plugin precisa conseguir “entregar” dados ao app. O agente deve implementar dois wrappers, sem refatorar.

### 5.1. Bridge para o Projeto (workspace)
Criar (se não existir) no app:

```python
def import_pattern_into_project(self, payload: dict) -> None:
    # payload deve conter pelo menos:
    # - cuts_2d: {'HRP': {...}, 'VRP': {...}} com angulos + mag_db/mag_lin + meta
    # - spherical_3d: {...} com theta_deg, phi_deg, mag_db/mag_lin
    # - meta: project/design/setup/sphere/freq/expr/timestamp
    #
    # 1) armazenar no estado atual (o mesmo que a UI já usa para plotar)
    # 2) chamar seus métodos já existentes de plot/update
    pass
```

**PASS:** depois de “Send to Project”, a UI mostra os novos diagramas (2D e referência do 3D).

### 5.2. Bridge para Library (SQLite)
Criar (se não existir) no gerenciador de banco:

```python
def add_diagram_entry(self, payload: dict) -> int:
    # Salva o dataset importado do AEDT na biblioteca.
    # Retorna ID do registro inserido.
    pass
```

**PASS:** depois de “Send to Library”, o item aparece na aba de biblioteca.

> Se o schema atual não suporta 3D, salvar 3D como:
- `npz` em disco + referência no DB (path + hash + meta)
- e 2D no formato já suportado (normalizado)

---

## 6) Teste REAL (obrigatório) — headless + UI

O agente deve executar ambos, anexar evidência e nunca “assumir” sucesso.

### 6.1. Teste headless: `tools/aedt_smoke_postproc.py` (o agente deve criar)
Criar `tools/aedt_smoke_postproc.py` com os passos:

1) Conectar ao AEDT 2025.2 (new session, non_graphical True)
2) Abrir `.aedt` de teste
3) Selecionar design HFSS
4) Garantir Infinite Sphere (ex.: `3D_Sphere`)
5) (Opcional) Rodar `analyze()`
6) Extrair:
   - HRP: Theta=90°, sweep Phi
   - VRP: Phi=0°, sweep Theta e converter para elevação
   - 3D: theta×phi (rota principal e fallback)
7) Exportar arquivos:
   - `out/hrp.json`
   - `out/vrp.json`
   - `out/ff_3d.npz`
   - `out/run_log.txt` (timings)

Execução:
```bash
python tools/aedt_smoke_postproc.py --aedt "D:\seu\projeto.aedt" --design "HFSSDesign1" --setup "Setup1 : LastAdaptive" --sphere "3D_Sphere" --out "out"
```

**PASS:** os 3 arquivos aparecem e têm shape coerente, sem NaN.

### 6.2. Teste UI (aba AEDT Live)
1) Abrir o AEDT 2025.2 e abrir o projeto HFSS com esfera.
2) Abrir o app.
3) Na aba AEDT Live:
   - escolher **Attach**
   - selecionar Project/Design/Setup/Sphere
   - clicar **Pull HRP** → plotar e habilitar “Send”
   - clicar **Pull VRP** → plotar e habilitar “Send”
   - clicar **Pull 3D** → exportar NPZ e/ou preencher viewer 3D
4) Executar:
   - Send to Project → atualiza workspace
   - Send to Library → cadastra na biblioteca

**PASS:** tudo acima funciona sem travar UI.

---

## 7) Padrões de pós-processamento (consistência com o app)

### 7.1. Definições
- HRP: Phi varre, Theta fixo em 90°
- VRP: Theta varre, Phi fixo em 0° (ou 90°)
- Conversão VRP para elevação do app:
  - `elev_deg = 90 - theta_deg`

### 7.2. Escala (campo vs potência)
HFSS “GainTotal” é em dB (potência). Para raio linear no 3D:
- Potência: `p_lin = 10**((db - db_max)/10)`
- Campo equivalente: `e_lin = 10**((db - db_max)/20)`

O agente deve deixar configurável na UI.

---

## 8) Performance e estabilidade (não negociável)

1) Nenhuma chamada AEDT pode rodar no thread da UI.
2) Cache:
   - chave por (project, design, setup, sphere, freq, expr, grid)
3) 3D:
   - geração do grid vetorizada (numpy), sem loops Python 361×361
   - export atômico (temp + replace)
4) Logs:
   - `logs/aedt_live.log` com timings por etapa:
     - connect, list, pull HRP/VRP/3D, send to library/project

---

## 9) Troubleshooting (casos comuns)

### 9.1. Não encontra sphere / far-field vazio
- Confirmar Infinite Sphere configurada no projeto.
- Implementar botão “Create Sphere” na aba AEDT Live (se faltar).

### 9.2. Dataset 3D não vem “multi-sweep”
- Usar fallback: varrer Phi e coletar Theta (mais lento, mas determinístico).
- Cachear resultado.

### 9.3. Erro de licença / AEDT não inicia
- Repetir smoke test do Desktop (seção 2.3).
- Corrigir license server/variáveis.

### 9.4. Só funciona no Python do AEDT
- Adotar modo recomendado: rodar o app com Python do AEDT
- ou criar microserviço local (fase 2) que retorna JSON/NPZ.

---

## 10) Entregáveis finais (o agente deve deixar pronto)

1) Aba AEDT Live integrada (sem alterar abas existentes).
2) `tools/aedt_smoke_postproc.py` funcionando.
3) Bridges de import para Library e Projeto.
4) Exports gerados em `output_dir`.
5) Logs e evidência do teste real.
