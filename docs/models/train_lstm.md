# Documentação: `src/models/train_lstm.py`

## O que este arquivo faz

Este módulo é o **script de treino offline** do modelo LSTM. Ele orquestra o fluxo completo: carregar CSV de gas, limpar e escalar features, criar sequências temporais, dividir em treino/validação/teste, treinar a LSTM e salvar modelo e scaler em disco.

### Funcionalidades principais

1. **`train_pipeline(data_path, model_dir, scaler_path)`**
   - Carrega o CSV com `load_raw_data` e limpa com `clean_data`.
   - Detecta automaticamente as colunas de feature e alvo:
     - Preferência para schema GoldRush: `gas_used`, `gas_limit`, `base_fee`, `tx_count`.
     - Fallback para formato legado: `avgGasPrice_Gwei`, `gasLimit`, `gasUsed`.
   - Escala as features com `scale_features`, cria sequências com `SequenceConfig(window_size=30, forecast_horizon=1)` e aplica `train_val_test_split`.
   - Constrói o modelo com `build_lstm_model`, treina com `train_lstm` (early stopping em val_loss).
   - Salva o modelo em `model_dir/lstm_gas_price.h5` e um dicionário em `scaler_path` (joblib) contendo: `scaler`, `feature_columns`, `sequence_config`.
   - Imprime loss de validação final e métricas MAE/MSE no conjunto de teste.

2. **CLI (`__main__`)**
   - Argumentos: `--data-path` (obrigatório), `--model-dir` (default: `models`), `--scaler-path` (default: `models/scaler.pkl`).
   - Chama `train_pipeline` com os valores informados.

### Dependências

- `src.features.build_features` (load_raw_data, clean_data, scale_features, create_sequences, train_val_test_split, SequenceConfig).
- `src.models.lstm_model` (build_lstm_model, train_lstm, save_model).
- `joblib` para persistir o scaler e a configuração.
- `tensorflow.keras.metrics` para MAE/MSE no teste.

---

## Importância no contexto do projeto

- **Ponto de entrada para treino via linha de comando**: permite treinar o modelo sem abrir o Jupyter, útil para rodar em servidor ou em pipelines (ex.: após gerar o CSV com `fetch_blocks_goldrush`).
- **Garante que modelo e scaler/config fiquem alinhados**: o mesmo scaler e `SequenceConfig` usados no treino são salvos e devem ser usados na inferência (ex.: notebook 03 ou uma futura API). Quem carrega o modelo deve carregar também o `scaler.pkl` para pré-processar os dados da mesma forma.
- **Suporte a múltiplos formatos de CSV**: a detecção de colunas GoldRush vs legado permite reutilizar o mesmo script para dados do `fetch_blocks_goldrush` ou de outras fontes (ex.: EthereumGasFee.csv com Date, Mean, Median, etc.), mantendo os notebooks capazes de “processar dados e treinar modelos” com diferentes datasets.
- **Papel na simplificação**: o projeto preserva “notebooks capazes de processar dados, treinar modelos e exibir gráficos”. Este script é a versão programática do treino; o notebook 02 replica o fluxo de forma narrativa. Ambos dependem de `build_features` e `lstm_model`.
