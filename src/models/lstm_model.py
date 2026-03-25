"""
Definição do modelo LSTM e rotinas de treino, salvamento e carregamento.

Este módulo abstrai a construção da rede recorrente usada para prever
as taxas de gas a partir das sequências geradas em `src/features/build_features.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from tensorflow import keras  # type: ignore[import]
from tensorflow.keras import layers  # type: ignore[import]

# Taxa padrão um pouco mais agressiva que 1e-3; ReduceLROnPlateau reduz se val_loss estagnar.
DEFAULT_ADAM_LR: float = 2.5e-3


def build_lstm_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
    *,
    learning_rate: float = DEFAULT_ADAM_LR,
) -> keras.Model:
    """
    Constrói um modelo LSTM simples para regressão de séries temporais.

    Parameters
    ----------
    input_shape:
        Tupla (window_size, n_features).
    units:
        Quantidade de neurônios na camada LSTM principal.
    dropout:
        Taxa de dropout entre camadas (0.0 a 1.0).
    learning_rate:
        Adam learning rate inicial.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units, return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int = 32,
    epochs: int = 1000,
) -> keras.callbacks.History:
    """
    Treina o modelo LSTM com early stopping e redução de LR em plateau de val_loss.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        min_delta=1e-6,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=5,
        min_lr=1e-6,
        verbose=0,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )
    return history


def save_model(model: keras.Model, path: str | Path) -> Path:
    """Salva o modelo Keras no caminho especificado."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return output_path


def load_model(path: str | Path) -> keras.Model:
    """Carrega um modelo previamente salvo (HDF5/.h5)."""
    # Keras 3+ não resolve "mse"/"mae" no HDF5; carregar sem compile e recompilar
    model = keras.models.load_model(path, compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_ADAM_LR),
        loss="mse",
        metrics=["mae"],
    )
    return model


def forecast_recursive_multistep(
    model: keras.Model,
    window_scaled: np.ndarray,
    *,
    target_feature_index: int,
    n_steps: int,
) -> np.ndarray:
    """
    Previsão autoregressiva: cada passo alimenta o valor predito de volta na janela.

    window_scaled deve ter shape (window_size, n_features) na mesma escala do treino.
    """
    w = np.asarray(window_scaled, dtype=np.float32).copy()
    out: list[float] = []
    for _ in range(n_steps):
        pred = model.predict(w[np.newaxis, :, :], verbose=0).reshape(-1)[0]
        out.append(float(pred))
        new_row = w[-1].copy()
        new_row[target_feature_index] = pred
        w = np.vstack([w[1:], new_row])
    return np.array(out, dtype=np.float64)


def forecast_sliding_onestep(
    model: keras.Model,
    scaled_values: np.ndarray,
    *,
    start_index: int,
    n_steps: int,
    window_size: int,
) -> np.ndarray:
    """
    Para cada dia t = start_index .. start_index + n_steps - 1, prevê y[t] usando apenas
    as linhas reais scaled_values[t - window_size : t] (um passo à frente, sem deriva).
    scaled_values: (n_timesteps, n_features), na escala do treino.
    """
    if start_index < window_size:
        raise ValueError(
            f"start_index ({start_index}) deve ser >= window_size ({window_size})"
        )
    last = start_index + n_steps
    if last > len(scaled_values):
        raise ValueError(
            "Série muito curta: precisa de linhas até índice "
            f"{last - 1}, mas há {len(scaled_values)}"
        )
    X_batch = np.stack(
        [
            scaled_values[start_index + k - window_size : start_index + k]
            for k in range(n_steps)
        ],
        axis=0,
    ).astype(np.float32)
    preds = model.predict(X_batch, verbose=0).reshape(-1)
    return np.asarray(preds, dtype=np.float64)


__all__ = [
    "DEFAULT_ADAM_LR",
    "build_lstm_model",
    "train_lstm",
    "save_model",
    "load_model",
    "forecast_recursive_multistep",
    "forecast_sliding_onestep",
]

