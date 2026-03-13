# Documentação: `src/features/build_features.py`

## O que este arquivo faz

Este módulo implementa o **pipeline de preparação de dados** para a LSTM: carregar CSV de gas, limpar, normalizar e transformar a série temporal em sequências (janelas) no formato esperado pela rede, além de dividir em treino/validação/teste.

### Funcionalidades principais

1. **`load_raw_data(path)`**
   - Lê o CSV e identifica a coluna de tempo (`signed_at` para GoldRush, ou `date`/`Date` para formatos legados).
   - Converte a coluna de tempo para datetime, ordena por ela e reindexa.

2. **`clean_data(df)`**
   - Ordena por tempo, remove linhas totalmente vazias, aplica forward fill e backward fill para valores faltantes.
   - Mantém a ordem temporal necessária para séries e sequências.

3. **`scale_features(df, feature_columns)`**
   - Aplica `MinMaxScaler` do scikit-learn às colunas numéricas indicadas.
   - Retorna o DataFrame com as colunas escaladas e o scaler treinado (para ser persistido e reutilizado na predição).

4. **`SequenceConfig` (dataclass)**
   - `target_column`: coluna alvo (ex.: `gas_used`).
   - `window_size`: tamanho da janela temporal (ex.: 30).
   - `forecast_horizon`: quantos passos à frente prever (ex.: 1).

5. **`create_sequences(df, feature_columns, config)`**
   - Constrói janelas de tamanho `window_size` sobre as features e o alvo deslocado por `forecast_horizon`.
   - Retorna `X` com shape `(n_amostras, window_size, n_features)` e `y` com shape `(n_amostras,)`.
   - Exige número mínimo de linhas (`window_size + forecast_horizon`); caso contrário levanta `ValueError` explicativo.

6. **`train_val_test_split(X, y, train_ratio, val_ratio)`**
   - Divide as sequências em treino, validação e teste **respeitando a ordem temporal** (sem embaralhar), para não vazar futuro no treino.

### Dependências

- `pandas`, `numpy`, `sklearn.preprocessing.MinMaxScaler`.

---

## Importância no contexto do projeto

- **Ponte entre dados brutos e modelo**: o CSV (gerado por `fetch_blocks_goldrush` ou externo) só pode ser usado pela LSTM após limpeza, escala e geração de sequências. Este módulo centraliza essa lógica.
- **Reutilizado em dois fluxos**: (1) no script `train_lstm.py` (CLI de treino) e (2) no notebook `02_treinamento_lstm.ipynb`. Manter a lógica aqui evita duplicação e garante consistência.
- **Compatibilidade com GoldRush e legado**: o suporte a colunas `signed_at` (GoldRush) e `date`/`Date` (e o uso de colunas como `gas_used`, `gas_limit`, `base_fee` ou `avgGasPrice_Gwei`, etc. em `train_lstm.py`) permite usar tanto o CSV do block_v2 quanto datasets em outros formatos.
- **Configuração de sequências**: `SequenceConfig` e os parâmetros de `create_sequences` definem a arquitetura temporal da rede (janela e horizonte); alterações aqui impactam diretamente o formato de entrada da LSTM em `lstm_model.py` e `train_lstm.py`.
