# Documentação: `src/models/lstm_model.py`

## O que este arquivo faz

Este módulo define a **arquitetura da rede LSTM** usada para previsão de métricas de gas (série temporal) e as rotinas de **treino, salvamento e carregamento** do modelo Keras.

### Funcionalidades principais

1. **`build_lstm_model(input_shape, units, dropout)`**
   - Constrói um `keras.Sequential` com:
     - `Input(shape=input_shape)` onde `input_shape = (window_size, n_features)`.
     - Uma camada LSTM com `units` neurônios (default 64), sem retorno de sequência.
     - Dropout (default 0.2).
     - Dense(32, relu) e Dense(1, linear) para saída de regressão.
   - Compila com Adam (lr=1e-3), loss MSE e métrica MAE.

2. **`train_lstm(model, X_train, y_train, X_val, y_val, batch_size, epochs)`**
   - Treina o modelo com `model.fit`, usando `validation_data=(X_val, y_val)`.
   - Usa callback `EarlyStopping` em `val_loss` com patience=5 e `restore_best_weights=True`.
   - Retorna o objeto `History` do Keras.

3. **`save_model(model, path)`**
   - Salva o modelo Keras no caminho indicado (ex.: `models/lstm_gas_price.h5`).
   - Cria o diretório pai se não existir.

4. **`load_model(path)`**
   - Carrega um modelo salvo (HDF5/.h5).
   - Como o Keras 3+ pode não resolver loss/metric no HDF5, carrega com `compile=False` e recompila com o mesmo optimizer, loss e métrica usados no treino.

### Dependências

- `tensorflow` (keras).

---

## Importância no contexto do projeto

- **Definição do modelo de predição**: toda a previsão de gas fees no projeto é feita por esta LSTM. O arquivo é o ponto único onde a arquitetura e hiperparâmetros (units, dropout, camadas) são definidos.
- **Integração com o pipeline**: `train_lstm.py` importa `build_lstm_model`, `train_lstm` e `save_model` para treinar e persistir o modelo; o notebook `02_treinamento_lstm.ipynb` usa as mesmas funções. O notebook `03_demonstracao_predicao.ipynb` (ou qualquer consumidor de modelo) usa `load_model` para carregar o `.h5` e fazer predições.
- **Consistência de treino e inferência**: o uso do mesmo `compile` no `load_model` garante que métricas e comportamento na carga sejam compatíveis com o treino.
- **Papel no fluxo simplificado**: o projeto foi simplificado para manter GET na Goldrush + notebooks (processar dados, treinar, gráficos). Este módulo é o núcleo da parte “treinar modelos” e “usar o modelo para exibir predições”.
