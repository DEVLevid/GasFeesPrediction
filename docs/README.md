# Documentação do projeto GasFeesPrediction

Cada arquivo de código ou notebook possui um arquivo de documentação relacionado que descreve **o que exatamente aquele arquivo faz** e **sua importância no contexto do projeto**.

## Mapeamento arquivo → documentação

| Arquivo | Documentação |
|---------|--------------|
| `src/api/goldrush_gas.py` | [docs/api/goldrush_gas.md](api/goldrush_gas.md) |
| `src/data/fetch_blocks_goldrush.py` | [docs/data/fetch_blocks_goldrush.md](data/fetch_blocks_goldrush.md) |
| `src/features/build_features.py` | [docs/features/build_features.md](features/build_features.md) |
| `src/models/lstm_model.py` | [docs/models/lstm_model.md](models/lstm_model.md) |
| `src/models/train_lstm.py` | [docs/models/train_lstm.md](models/train_lstm.md) |
| `notebooks/01_exploracao_dados.ipynb` | [docs/notebooks/01_exploracao_dados.md](notebooks/01_exploracao_dados.md) |
| `notebooks/02_treinamento_lstm.ipynb` | [docs/notebooks/02_treinamento_lstm.md](notebooks/02_treinamento_lstm.md) |
| `notebooks/03_demonstracao_predicao.ipynb` | [docs/notebooks/03_demonstracao_predicao.md](notebooks/03_demonstracao_predicao.md) |
| `notebooks/04_predicao_2025_mensal.ipynb` | Predição mensal 2025 com LSTM e dados reais via API GoldRush (sem doc dedicado ainda). |

Cada documento contém:
- **O que o arquivo faz**: funções principais, entradas/saídas, dependências e variáveis de ambiente (quando aplicável).
- **Importância no contexto do projeto**: papel no fluxo de dados, relação com outros módulos e com os objetivos do projeto (GET Goldrush, processamento, treino, gráficos de predição).
