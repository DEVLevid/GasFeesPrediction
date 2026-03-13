# Documentação: `src/api/goldrush_gas.py`

## O que este arquivo faz

Este módulo é a **única interface do projeto com o endpoint de gas prices da API GoldRush (Covalent)**. Ele realiza chamadas GET para obter os preços de gas em tempo real para uma chain configurada.

### Funcionalidades principais

1. **`get_gas_prices(chain_name, event_type, quote_currency)`**
   - Faz uma requisição GET para `{base_url}/v1/{chain_name}/event/{event_type}/gas_prices`.
   - Retorna um `pd.DataFrame` com uma linha por faixa de gas (items) retornada pela API, além de metadados (chain_id, chain_name, updated_at, base_fee, etc.).
   - Parâmetros podem ser omitidos e são lidos do `.env` (GOLDRUSH_CHAIN_NAME, GOLDRUSH_EVENT_TYPE, GOLDRUSH_QUOTE_CURRENCY).
   - Trata erros 401 (credencial inválida) e resposta malformada; em resposta vazia retorna DataFrame vazio.
   - Adiciona coluna `timestamp_utc` com o momento da coleta local.

2. **`get_current_gas_snapshot(chain_name, event_type)`**
   - Utiliza `get_gas_prices()` e retorna apenas a primeira linha (uma única linha com o estado atual de gas).
   - Útil para quem precisa de um único snapshot em vez de todas as faixas.

3. **Autenticação**
   - Função interna `_get_auth_header()` monta o header `Authorization: Bearer <GOLDRUSH_API_KEY>`.
   - A chave é obrigatória; se não estiver definida, as funções públicas levantam `RuntimeError`.

### Dependências

- `requests` para HTTP.
- `pandas` para o retorno em DataFrame.
- `python-dotenv` para carregar variáveis do `.env`.

### Variáveis de ambiente utilizadas

| Variável | Uso |
|----------|-----|
| `GOLDRUSH_API_KEY` | Token Bearer (obrigatório). |
| `GOLDRUSH_BASE_URL` | Base da API (default: https://api.covalenthq.com). |
| `GOLDRUSH_CHAIN_NAME` | Nome da chain (ex.: eth-mainnet). |
| `GOLDRUSH_EVENT_TYPE` | Tipo de evento (ex.: erc20). |
| `GOLDRUSH_QUOTE_CURRENCY` | Moeda de cotação (ex.: USD). |

---

## Importância no contexto do projeto

- **Fonte única de “preços de gas em tempo real”**: todo o projeto que precisa consultar os preços atuais de gas na GoldRush deve usar este módulo. Evita duplicação de lógica de URL, autenticação e parsing.
- **Alinhado ao objetivo do projeto**: o GasFeesPrediction foi simplificado para preservar (1) o GET na API Goldrush para buscar preços de gas e (2) os notebooks de processamento, treino e gráficos. Este arquivo é o núcleo do item (1).
- **Não substitui dados históricos**: para treinar a LSTM são usados dados por bloco (gas_used, base_fee, etc.) obtidos por `fetch_blocks_goldrush.py` (endpoint block_v2). O `goldrush_gas.py` atende consultas pontuais ou em tempo real aos preços de gas (endpoint gas_prices).
