# Documentação: `notebooks/02_treinamento_lstm.ipynb`

## O que este notebook faz

Este notebook ilustra o **pipeline completo de treino** da LSTM para previsão de taxas de gas. Ele cobre, em formato narrativo e executável:

1. **Geração ou seleção do dataset**
   - Se `INPUT_CSV` for `None`: gera dados via GoldRush (block_v2) usando `fetch_blockchain_data`, `date_to_block_range_hours` ou `_date_to_block_range` (ex.: 5 dias ou 1 hora na data de ontem) e salva em CSV.
   - Se `INPUT_CSV` for um caminho: usa esse CSV existente (ex.: `data/gas_data_tcc.csv` ou um CSV externo como EthereumGasFee.csv).

2. **Preparação dos dados**
   - Carrega o CSV com `load_raw_data`, limpa com `clean_data`, escala com `scale_features`, cria sequências com `create_sequences` e `SequenceConfig`, e divide com `train_val_test_split` (módulos de `src.features.build_features`).

3. **Treino**
   - Constrói o modelo com `build_lstm_model`, treina com `train_lstm` (early stopping) e salva modelo e scaler com `save_model` e joblib (módulos de `src.models.lstm_model`).

4. **Visualização**
   - Gráficos de valor real vs predito (treino/validação ou teste) e curva de loss no treino.

### Dependências do notebook

- `src.data.fetch_blocks_goldrush`: `_date_to_block_range`, `date_to_block_range_hours`, `fetch_blockchain_data`.
- `src.features.build_features`: `SequenceConfig`, `clean_data`, `create_sequences`, `load_raw_data`, `scale_features`, `train_val_test_split`.
- `src.models.lstm_model`: `build_lstm_model`, `save_model`, `train_lstm`.
- joblib, numpy, pandas, matplotlib, pathlib.

---

## Importância no contexto do projeto

- **Reprodução do treino passo a passo**: permite executar e inspecionar cada etapa (dados, sequências, treino, métricas) no Jupyter, em contraste com o script `train_lstm.py`, que executa o pipeline de uma vez pela CLI.
- **Flexibilidade de entrada**: suporta tanto “gerar dados pela API GoldRush” quanto “usar CSV já existente”, mantendo o projeto capaz de processar dados e treinar modelos em diferentes cenários.
- **Documentação do fluxo**: mostra claramente como os módulos `fetch_blocks_goldrush`, `build_features` e `lstm_model` se encaixam; útil para TCC e onboarding.
- **Saída para o notebook 03**: produz o modelo (`models/lstm_gas_price.h5`) e o scaler/config (`models/scaler.pkl`) que o notebook de demonstração de predição carrega para exibir gráficos de predição vs real.
