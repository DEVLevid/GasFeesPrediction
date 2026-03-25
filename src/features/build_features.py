"""
Rotinas de preparação de dados e criação de sequências temporais para a LSTM.

Fluxo principal:
- Carregar dados brutos a partir de um CSV consolidado.
- Limpar, ordenar e tratar valores faltantes.
- Normalizar/escala as features numéricas.
- Transformar a série temporal em janelas (sequências) adequadas para a LSTM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import]


@dataclass
class SequenceConfig:
    """Configuração para criação de sequências temporais."""

    target_column: str = "gas_used"
    window_size: int = 30
    forecast_horizon: int = 1


def _time_column(df: pd.DataFrame) -> str | None:
    """Retorna a coluna de tempo disponível: signed_at (GoldRush), date ou Date."""
    if "signed_at" in df.columns:
        return "signed_at"
    if "date" in df.columns:
        return "date"
    if "Date" in df.columns:
        return "Date"
    return None


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV de gas (GoldRush block_v2 ou outro pipeline)."""
    df = pd.read_csv(path)
    time_col = _time_column(df)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df.sort_values(time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza simples: ordenar por tempo, remover linhas vazias, forward/back fill.
    Suporta coluna de tempo 'signed_at' (GoldRush) ou 'date'.
    """
    result = df.copy()
    time_col = _time_column(result)
    if time_col:
        result[time_col] = pd.to_datetime(result[time_col], errors="coerce")
        result.sort_values(time_col, inplace=True)

    result.dropna(how="all", inplace=True)
    result.ffill(inplace=True)
    result.bfill(inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def scale_features(
    df: pd.DataFrame, feature_columns: Iterable[str]
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Aplica MinMaxScaler às colunas numéricas selecionadas.

    Retorna o DataFrame transformado e o scaler treinado.
    """
    scaler = MinMaxScaler()
    features = df[list(feature_columns)].astype(float)
    scaled_values = scaler.fit_transform(features)

    df_scaled = df.copy()
    df_scaled[list(feature_columns)] = scaled_values
    return df_scaled, scaler


def minmax_transform_clip(
    scaler: MinMaxScaler,
    X: np.ndarray | pd.DataFrame,
    *,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """
    `transform` seguido de clip — útil na inferência quando o período de teste tem
    valores acima do máximo (ou abaixo do mínimo) visto no treino: sem o clip, a LSTM
    recebe entradas fora de [0, 1] e tende a degradar fortemente.
    """
    arr = np.asarray(scaler.transform(X), dtype=np.float64)
    return np.clip(arr, clip_min, clip_max)


def create_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências (janelas) de dados para alimentar a LSTM.

    Parameters
    ----------
    df:
        DataFrame já limpo e, preferencialmente, com features escaladas.
    feature_columns:
        Lista de colunas usadas como entrada da rede.
    config:
        Configuração com nome da coluna alvo, tamanho da janela e horizonte de previsão.

    Returns
    -------
    X, y:
        - X com shape (n_amostras, window_size, n_features)
        - y com shape (n_amostras,)
    """
    values = df[feature_columns].values
    target = df[config.target_column].values

    window = config.window_size
    horizon = config.forecast_horizon

    indices = range(window, len(df) - horizon + 1)
    seq_list = [values[end_idx - window : end_idx] for end_idx in indices]
    y_list = [float(target[end_idx - 1 + horizon]) for end_idx in indices]

    if not seq_list:
        min_required = window + horizon
        raise ValueError(
            f"Dados insuficientes para criar sequências com a configuração atual. "
            f"Precisa de pelo menos {min_required} linhas (window_size={window} + forecast_horizon={horizon}), "
            f"mas o DataFrame tem {len(df)} linhas. Reduza window_size ou use um dataset maior."
        )

    X = np.array(seq_list)
    y = np.array(y_list)
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide as sequências em conjuntos de treino, validação e teste respeitando a ordem temporal.
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


__all__ = [
    "SequenceConfig",
    "load_raw_data",
    "clean_data",
    "scale_features",
    "minmax_transform_clip",
    "create_sequences",
    "train_val_test_split",
]

