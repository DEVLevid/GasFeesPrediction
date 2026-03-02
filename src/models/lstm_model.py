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


def build_lstm_model(
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
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
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
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
    epochs: int = 50,
) -> keras.callbacks.History:
    """
    Treina o modelo LSTM com early stopping baseado na perda de validação.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
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
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


__all__ = ["build_lstm_model", "train_lstm", "save_model", "load_model"]

