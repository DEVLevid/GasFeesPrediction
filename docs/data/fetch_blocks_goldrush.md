# Documentação: `src/data/fetch_blocks_goldrush.py`

## O que este arquivo faz

Este módulo **minera dados históricos de blocos** via API GoldRush (Covalent), usando o endpoint **block_v2** (`GET /v1/{chainName}/block_v2/{blockHeight}/`). O resultado é um dataset em CSV usado para treinar o modelo LSTM de predição de gas fees.

### Funcionalidades principais

1. **`fetch_blockchain_data(start_block, end_block, api_key, chain_name, include_tx_count, step=1)`**
   - Itera sobre blocos no intervalo `[start_block, end_block]`. O parâmetro **step** (default 1) amostra um bloco a cada `step` (ex.: `step=7200` para ~1 bloco por dia na Ethereum), reduzindo requisições para períodos longos (ex.: ano 2025).
   - Para cada bloco chama o endpoint block_v2 e extrai: `block_height`, `signed_at`, `gas_used`, `gas_limit`, `base_fee`. Opcionalmente obtém `tx_count` via endpoint de transações (transactions_v3) quando `include_tx_count=True`.
   - Em falha (conexão, 429 rate limit, resposta vazia) a linha é preenchida com NaN e o loop continua (resiliência).
   - Usa intervalo entre requisições (`REQUEST_INTERVAL_SEC`) e backoff em caso de 429, com retentativas limitadas (`MAX_RETRIES`).

2. **Conversão data → blocos**
   - `_date_to_block_range(start_date, end_date, chain_name)`: converte intervalo de datas (YYYY-MM-DD) em intervalo de blocos aproximado, usando referência fixa (bloco conhecido em data UTC) e blocos/dia por chain (Ethereum ~7200/dia, Polygon ~43200/dia).
   - `date_to_block_range_hours(start_date, hours, chain_name)`: converte uma data e duração em horas em intervalo de blocos; útil para datasets de teste (ex.: 1 hora).

3. **CLI (`main`)**
   - Permite informar bloco inicial e final, ou `--start-date` e `--end-date`, ou `--start-date` com `--hours`, ou `--yesterday` com `--hours`.
   - Opções: `--chain`, `--tx-count`, `-o/--output` para o CSV.
   - Gera o CSV (ex.: `data/gas_data_tcc.csv`) com as colunas citadas.

### Dependências

- `requests`, `pandas`, `tqdm`, `python-dotenv`.

### Variáveis de ambiente

- `GOLDRUSH_API_KEY`, `GOLDRUSH_BASE_URL`, `GOLDRUSH_CHAIN_NAME` (fallback da CLI).

---

## Importância no contexto do projeto

- **Fonte dos dados de treino**: o pipeline de ML (build_features → train_lstm) e os notebooks dependem de um CSV com métricas por bloco. Esse CSV é gerado por este script. Sem ele, seria necessário obter dados históricos por outro meio.
- **Complementar ao `goldrush_gas.py`**: `goldrush_gas.py` fornece preços de gas em tempo real (endpoint gas_prices). Este arquivo fornece dados históricos por bloco (block_v2), necessários para treinar e avaliar a LSTM.
- **Uso nos notebooks**: o notebook `02_treinamento_lstm.ipynb` pode gerar o dataset chamando `fetch_blockchain_data` e helpers (`date_to_block_range_hours`, `_date_to_block_range`) quando não há CSV pronto; caso contrário usa um CSV existente (ex.: de uma execução prévia deste script).
- **Resiliência**: o tratamento de rate limit e falhas permite coletar grandes intervalos sem parar o processo por um único bloco falho.
