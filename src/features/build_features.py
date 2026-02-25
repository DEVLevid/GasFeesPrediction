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

    target_column: str = "avgGasPrice_Gwei"
    window_size: int = 30
    forecast_horizon: int = 1


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV consolidado de gas gerado pelo pipeline de coleta."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza simples dos dados:
    - Ordenar por data.
    - Remover linhas completamente vazias.
    - Forward-fill de valores faltantes onde fizer sentido.
    """
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"])
        result.sort_values("date", inplace=True)

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
        raise ValueError("Dados insuficientes para criar sequências com a configuração atual.")

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
    "create_sequences",
    "train_val_test_split",
]

