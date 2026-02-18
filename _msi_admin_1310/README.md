# Antenna Pattern Converter & Library (EFTX)

Aplicativo desktop desenvolvido em Python com CustomTkinter para gerenciar, visualizar, converter e exportar diagramas de irradiação de antenas.

![Logo](eftx-logo.png)

## Funcionalidades

### 1. Visualização e Conversão
- **Importação**: Suporta arquivos .CSV (formato HFSS), .TXT, .PAT e .PRN.
- **Gráficos**: Visualização simultânea Planar (VRP) e Polar (HRP).
- **Exportação**:
  - **.PAT (NSMA)**: Formato padrão.
  - **.PAT (ADT)**: Formato RFS Voltage.
  - **.PRN**: Arquivo combinado VRP+HRP com metadados.

### 2. Biblioteca de Diagramas (Batch)
- Persistência local via SQLite (`library.db`).
- Visualização em grade responsiva com miniaturas.
- Filtros e carregamento rápido para a área de trabalho.

### 3. Composição de Arrays
- Simulação de arrays verticais e painéis horizontais.
- Ajuste de tilt, null fill e número de elementos.

## Instalação

Necessário Python 3.8+.

1. Clone o repositório:
   ```bash
   git clone https://github.com/Gecesars/antenna_prn_pat_generator.git
   ```

2. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Execução

```bash
python deep3.py
```

## Estrutura do Projeto

- `deep3.py`: Código fonte principal da aplicação.
- `library.db`: Banco de dados SQLite da biblioteca de diagramas (gerado localmente).
- `eftx-logo.png`: Logo da aplicação.

## Autoria

Desenvolvido para EFTX Broadcast Television & Radio.
