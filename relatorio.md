# EFTX Diagram Suite — Exportação de Relatório PDF (Instrução para o Agente)

> **Objetivo**  
Implementar a exportação de um **relatório PDF multipágina** com **1 diagrama + 1 tabela + métricas por página**, gerado a partir dos dados do projeto/diagramas já carregados na aplicação, usando como **template de fundo** o arquivo PDF fornecido (anexo do projeto).

**Template (underlay) a usar:** `./modelo.pdf

---

## 0) Regras rígidas

1. **Não alterar** fluxos existentes de importação, composição, cálculo e exportação atuais (PRN/PAT/PNG/CSV etc.).  
2. O relatório PDF deve ser uma **nova funcionalidade** (“Exportar Relatório PDF”).  
3. Cada página deve conter:
   - **Título** do corte (coerente e padronizado)
   - **Diagrama** (imagem do gráfico)
   - **Bloco de métricas** (resumo numérico)
   - **Tabela** do corte (amostrada/compactada para caber, mas com opção de salvar tabela completa separadamente)
4. **Não abrir janela modal** para o relatório. Usar diálogo de arquivo “Salvar como…” e logs/progresso na UI.
5. Respeitar **margens** e **área segura** do template (evitar sobreposição de cabeçalho/rodapé do template).
6. **Thread/worker obrigatório**: o export não pode travar a UI.

---

## 1) Dependências

Adicionar ao ambiente (requirements/pyproject):
- `reportlab` — geração do PDF de conteúdo
- `pypdf` — mesclagem do template como **underlay** em cada página
- `Pillow` (opcional) — validação/tuning de imagens (se necessário)

---

## 2) Arquitetura sugerida (módulo isolado)

Criar um módulo dedicado para não contaminar o restante da base:

```
app/
  reports/
    __init__.py
    pdf_report.py     # API pública export_report_pdf(...)
    layout.py         # constantes (A4, margens, estilos, tamanhos)
    tables.py         # render da tabela + compactação + paginação
    metrics.py        # montagem do bloco de métricas (formatado)
    merge_template.py # merge underlay com pypdf
  assets/
    templates/
      eftx_report_template.pdf   # copiar o template anexado aqui
```

**Regra:** `pdf_report.py` deve ser o único ponto de entrada para a UI.

---

## 3) Fonte de dados (contrato interno)

O exportador deve receber um **payload normalizado**. Não “puxar” estado global solto.

### 3.1 Estrutura mínima por página (corte)
Para cada página/corte, o exportador precisa:

- `page_title`: string (ex.: `"Diagrama AZ_POL1 (Azimute)"`)
- `page_subtitle`: string curta (projeto, antena/base, pol, freq, expressão)
- `plot_image_path`: caminho do PNG do diagrama (alta resolução)
- `metrics`: dict com chaves/valores já em unidades corretas (ex.: pico, HPBW, F/B, faixa, passo, pontos, etc.)
- `table`: estrutura tabular (lista de linhas) ou arrays (ângulo, valor) com metadados:
  - `columns`: ex.: `["Ang [deg]", "Valor [dB]"]` (ou 4 colunas para tabela dupla)
  - `rows`: lista de linhas
  - `note`: texto curto (ex.: “Tabela exibida amostrada; CSV contém dados completos”, quando aplicável)

### 3.2 Como gerar `plot_image_path`
**Obrigatório:** gerar o PNG do diagrama em **alta resolução** antes do PDF.

Recomendação:
- `dpi >= 200` (ideal 300 para impressão)
- tamanho consistente (ex.: largura ~ 1600–2400 px)
- fundo branco (para relatório) ou adaptável (se o template for escuro, ajustar)
- preservar proporção do polar/cartesiano

**Importante:** nunca reusar screenshots da UI. Sempre renderizar via Matplotlib no backend.

### 3.3 Como gerar a tabela para o PDF
A tabela exibida no PDF deve caber na página. Para cortes com milhares de pontos:
- gerar **CSV completo** em paralelo (sem amostragem)
- aplicar **amostragem de exibição** (somente para o PDF) para limitar linhas

Regras de amostragem (simples, robustas):
- limite de linhas por página: **60 a 90**
- sempre incluir:
  - ponto de pico
  - mínimos relevantes (menor valor)
  - pontos próximos a métricas importantes já calculadas (ex.: ângulos de largura de feixe)
  - bordas da faixa angular
- completar com amostragem uniforme do restante

---

## 4) Layout (A4, margens, área segura)

### 4.1 Página
- **Formato:** A4 retrato (210 × 297 mm)
- **Template:** o PDF template deve ser o **underlay** (fundo) em todas as páginas.

### 4.2 Área segura (safe box)
Definir uma área segura fixa para desenhar conteúdo, em `layout.py`:

**Recomendado (inicial):**
- `left = 24 mm`
- `right = 18 mm`
- `top = 38 mm`
- `bottom = 30 mm`

> Ajustar apenas estes valores caso algum elemento do template colida com o conteúdo.

### 4.3 Grade interna (1 diagrama + 1 tabela + métricas)
Dentro da área segura:
- **Topo:** título (central) + subtítulo (menor)
- **Meio:** diagrama grande
- **Bloco de métricas:** ao lado do diagrama **ou** logo abaixo do subtítulo (se o diagrama ocupar a largura toda)
- **Base:** tabela (zebra, cabeçalho, alinhamento numérico)

Regras:
- Diagrama deve manter proporção e ser legível (não “espremido”)
- Tabela deve ter tipografia menor (8–9 pt) e cabeçalho repetido
- Nunca desenhar texto “em cima” do template (respeitar safe box)

---

## 5) Títulos e nomenclatura (coerência)

Padronizar títulos para todas as páginas:

- **Título (H1):** `Diagrama {NOME_DO_CORTE} ({PLANO})`
- **Subtítulo (H2):** `Projeto: {PROJECT} • Antena: {BASE_NAME} • Pol: {POL} • Freq: {FREQ} MHz • Expr: {EXPR}`

Regras:
- Título sempre **centralizado**
- Subtítulo com alinhamento consistente
- Evitar abreviações variáveis → escolher uma e manter

---

## 6) Renderização do PDF (pipeline recomendado)

Implementar em duas etapas para máxima compatibilidade:

### Etapa A — gerar PDF de conteúdo (sem fundo)
Usar `reportlab` (Platypus):
- 1 página por corte
- inserir título, subtítulo, imagem do diagrama, métricas e tabela

Salvar como: `report_content.pdf` (temporário no diretório de saída)

### Etapa B — aplicar template como underlay
Usar `pypdf`:
- abrir `eftx_report_template.pdf`
- para cada página `i`, fazer merge com o conteúdo
- salvar como `relatorio.pdf` final

> Garantir a ordem correta (template embaixo, conteúdo em cima) e validar visualmente.

---

## 7) Formatação profissional das tabelas

Implementar em `tables.py`:

- Cabeçalho com fundo discreto e fonte em negrito
- Corpo com fonte 8–9 pt
- Linhas com zebra (cinza muito claro alternado)
- Bordas finas e discretas
- Valores numéricos alinhados à direita
- Opção de tabela em **duas colunas** (Ang/Valor | Ang/Valor) para aproveitar a largura

Se não couber:
- reduzir número de linhas exibidas (amostragem)
- inserir nota: “Tabela exibida compactada; dados completos no CSV”

---

## 8) Bloco de métricas (compacto e legível)

Implementar em `metrics.py`:

- Exibir em formato “rótulo : valor”
- Unidades explícitas
- Arredondamento coerente (ex.: 2 casas em dB; 2 em graus)
- Distribuir em 2 colunas se houver muitos itens
- No máximo ~10–14 linhas

---

## 9) Integração na UI (sem modal)

### 9.1 Menu
Adicionar:
- `Arquivo → Exportar → Relatório PDF…`

### 9.2 Diálogo
Ao clicar:
- escolher nome/caminho do PDF (default: `{BaseName}_relatorio.pdf`)
- opções:
  - quais cortes incluir (AZ/EL, POL1/POL2, VRP/HRP)
  - ordem das páginas
  - DPI da imagem (200/300)
  - “Salvar tabelas completas (CSV) junto do PDF” (default ON)

### 9.3 Execução em worker
- rodar export em thread/worker
- atualizar barra de progresso + log incremental (“Gerando página 3/8: EL_POL1…”)
- permitir cancelamento limpo (se existir token de cancelamento)

---

## 10) Logs e validações

Durante a geração:
- validar que a imagem existe e tem resolução mínima
- validar que há métricas essenciais (pelo menos pico e faixa/pontos)
- validar que a tabela tem linhas

Ao final:
- logar caminho final do PDF
- logar caminhos dos CSVs (se exportados)
- logar warnings:
  - tabela compactada
  - imagem re-renderizada por falta de DPI
  - template não encontrado (falha dura)

---

## 11) Testes obrigatórios (o agente deve executar de verdade)

1. **Teste 1 página:** gerar PDF com 1 corte e verificar que:
   - títulos não colidem com cabeçalho do template
   - tabela não colide com rodapé
2. **Teste lote (8 páginas):** AZ/EL × POL1/POL2 e validar consistência visual
3. **Teste tabela grande:** corte com muitos pontos
   - confirmar amostragem no PDF
   - confirmar CSV completo salvo
   - confirmar log indicando compactação
4. **Teste regressão:** salvar um PDF “golden” e comparar visualmente após ajustes

---

## 12) Critérios de aceite (Definition of Done)

- Export gera multipágina com template aplicado em todas as páginas.
- 1 diagrama + 1 tabela + métricas por página, layout consistente.
- Títulos coerentes e precisos.
- Margens respeitadas (sem sobreposição com o template).
- UI não trava; logs e progresso funcionam.
- Tabelas legíveis e bem formatadas; CSV completo opcional salvo.
- Testes executados e registrados.

---

## 13) Observações finais

- Centralizar constantes de layout em `layout.py` (um único ponto de ajuste).
- Evitar dimensões “mágicas” espalhadas; tudo parametrizado.
- Copiar o template para `app/assets/templates/eftx_report_template.pdf` e versionar no repositório.
