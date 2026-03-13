# Documentação: `notebooks/03_demonstracao_predicao.ipynb`

## O que este notebook faz

Este é o **notebook de demonstração visual** da predição de taxas de gas. Ele carrega os artefatos produzidos no fluxo de treino e permite comparar, em um período escolhido, a série histórica real com as predições do modelo LSTM.

### Conteúdo típico

- **Carregamento de artefatos**:
  - Dataset consolidado de gas (ex.: CSV usado no treino ou com o mesmo schema).
  - Modelo LSTM treinado (`models/lstm_gas_price.h5`).
  - Scaler e configuração (`models/scaler.pkl`), para pré-processar os dados da mesma forma que no treino.

- **Seleção de período**: definição do intervalo de análise (datas ou índices) sobre o qual serão exibidos histórico e predições.

- **Comparação visual**:
  - Série histórica real (ex.: `gas_used` ou `base_fee` do dataset GoldRush).
  - Predições do modelo para esse período.
  - Valores reais correspondentes, para avaliar a qualidade da predição (erro, tendência).

O notebook pode usar apenas pandas, matplotlib (ou plotly), joblib e o módulo de carregamento do modelo (ex.: `src.models.lstm_model.load_model` ou equivalente) sem depender dos outros módulos de `src` para a parte de dados, desde que o CSV e o scaler estejam no formato esperado.

---

## Importância no contexto do projeto

- **Demonstração do valor do modelo**: mostra na prática o resultado do pipeline (dados → features → LSTM → predições) e permite comunicar os resultados do TCC ou para stakeholders.
- **Validação qualitativa**: além das métricas (MAE/MSE) do script de treino, os gráficos permitem inspecionar se o modelo acompanha picos, tendências e sazonalidade.
- **Base para integração futura**: o plano do projeto menciona possível integração com aplicação web (API que retorna histórico/predições, frontend com gráficos). Este notebook é o protótipo do fluxo “carregar modelo + dados → gerar predições → exibir gráficos”.
- **Fechamento do fluxo simplificado**: o projeto foi simplificado para manter (1) GET na Goldrush para preços de gas e (2) notebooks capazes de processar dados, treinar modelos e exibir gráficos de predição. Este notebook cumpre a parte “exibir gráficos de predição” e depende diretamente dos outputs do 02 (modelo e scaler).
