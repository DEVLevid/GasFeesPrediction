# Documentação: `notebooks/01_exploracao_dados.ipynb`

## O que este notebook faz

Este é o **notebook de exploração inicial** do dataset de gas. Ele carrega um CSV de gas (em geral gerado pela API GoldRush via `fetch_blocks_goldrush`, por exemplo `data/gas_data_tcc.csv`) e realiza análises descritivas para entender a qualidade e o comportamento dos dados antes do treino do modelo.

### Conteúdo típico

- **Carregamento do dataset**: leitura do CSV e verificação de colunas (ex.: `block_height`, `signed_at`, `gas_used`, `gas_limit`, `base_fee`, `tx_count`).
- **Estatísticas básicas**: resumo das métricas (média, desvio, min, max, etc.) para avaliar distribuição e possíveis outliers.
- **Séries temporais**: gráficos das principais variáveis ao longo do tempo (ex.: gas_used, base_fee por data/bloco) para visualizar tendências e sazonalidade.
- **Correlação**: análise de correlação entre variáveis (heatmap ou tabela) para entender relações que o modelo LSTM pode explorar.

O notebook **não importa módulos de `src`** obrigatoriamente; pode usar apenas pandas, matplotlib/seaborn e numpy para as análises. O caminho do CSV é configurado na própria célula (ex.: `data/gas_data_tcc.csv`).

---

## Importância no contexto do projeto

- **Primeiro passo do fluxo narrativo**: no fluxo “exploração → treino → demonstração de predição”, este notebook cobre a exploração. Ajuda a validar que o dataset está completo e coerente antes de gastar tempo no treino.
- **Documentação viva dos dados**: deixa explícito qual é a origem dos dados (GoldRush, script `fetch_blocks_goldrush`) e quais colunas são usadas no projeto, facilitando reprodução e TCC.
- **Base para decisões de feature engineering**: as correlações e séries temporais podem orientar a escolha de colunas em `build_features` e em `train_lstm` (quais features incluir no treino).
- **Alinhado à simplificação**: o projeto foi simplificado para manter os notebooks capazes de processar dados, treinar modelos e exibir gráficos. Este notebook cumpre a parte de “processar/entender os dados” e prepara o terreno para o 02 (treino) e 03 (gráficos de predição).
