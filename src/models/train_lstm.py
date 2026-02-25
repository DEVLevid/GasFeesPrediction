"""
Script de treino offline do modelo LSTM.

Fluxo:
- Carrega o CSV consolidado de gas.
- Limpa e escala as features.
- Cria sequências temporais.
- Divide em treino/validação/teste.
- Treina o modelo LSTM e salva em disco.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.features.build_features import (
    SequenceConfig,
    clean_data,
    create_sequences,
    load_raw_data,
    scale_features,
    train_val_test_split,
)
from src.models.lstm_model import build_lstm_model, save_model, train_lstm


def train_pipeline(
    data_path: str | Path,
    model_dir: str | Path = "models",
    scaler_path: str | Path = "models/scaler.pkl",
) -> None:
    """
    Executa o pipeline completo de treino da LSTM a partir de um CSV consolidado.
    """
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    scaler_path = Path(scaler_path)

    df = load_raw_data(data_path)
    df = clean_data(df)

    # Features de entrada e coluna alvo principais.
    feature_columns = []
    if "avgGasPrice_Gwei" in df.columns:
        feature_columns.append("avgGasPrice_Gwei")
    if "gasLimit" in df.columns:
        feature_columns.append("gasLimit")
    if "gasUsed" in df.columns:
        feature_columns.append("gasUsed")

    if not feature_columns:
        raise RuntimeError(
            "Nenhuma coluna de feature encontrada (esperado: avgGasPrice_Gwei, gasLimit, gasUsed)."
        )

    df_scaled, scaler = scale_features(df, feature_columns=feature_columns)

    config = SequenceConfig(
        target_column="avgGasPrice_Gwei",
        window_size=30,
        forecast_horizon=1,
    )

    X, y = create_sequences(df_scaled, feature_columns=feature_columns, config=config)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_lstm(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=32,
        epochs=50,
    )

    print("Treino concluído. Última loss de validação:", history.history["val_loss"][-1])

    model_path = model_dir / "lstm_gas_price.h5"
    save_model(model, model_path)
    print(f"Modelo salvo em: {model_path}")

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "sequence_config": config,
        },
        scaler_path,
    )
    print(f"Scaler e configuração salvos em: {scaler_path}")

    # Opcional: simples avaliação em teste (MSE/MAE).
    from tensorflow.keras.metrics import (  # type: ignore[import]
        MeanAbsoluteError,
        MeanSquaredError,
    )

    y_pred = model.predict(X_test, verbose=0).squeeze()
    mae = MeanAbsoluteError()
    mse = MeanSquaredError()
    mae.update_state(y_test, y_pred)
    mse.update_state(y_test, y_pred)
    print(f"MAE teste: {float(mae.result()):.6f}, MSE teste: {float(mse.result()):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina um modelo LSTM para previsão de taxas de gas."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Caminho para o CSV consolidado de gas (gerado por fetch_gas_data).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Diretório onde o modelo será salvo.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/scaler.pkl",
        help="Caminho para salvar o scaler e a configuração.",
    )

    args = parser.parse_args()

    train_pipeline(
        data_path=args.data_path,
        model_dir=args.model_dir,
        scaler_path=args.scaler_path,
    )

