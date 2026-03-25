"""Pacote de modelos (expõe LSTM)."""

from .lstm_model import (
    DEFAULT_ADAM_LR,
    build_lstm_model,
    forecast_recursive_multistep,
    forecast_sliding_onestep,
    load_model,
    save_model,
    train_lstm,
)

__all__ = [
    "DEFAULT_ADAM_LR",
    "build_lstm_model",
    "forecast_recursive_multistep",
    "forecast_sliding_onestep",
    "load_model",
    "save_model",
    "train_lstm",
]
