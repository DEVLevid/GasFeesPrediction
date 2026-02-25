"""
Coletor contínuo (ou pontual) do gas price atual usando a API GoldRush (Covalent).

Anexa as leituras em um CSV que pode ser usado como dataset para o treino da LSTM.
Configuração via variáveis de ambiente: GOLDRUSH_API_KEY, GOLDRUSH_CHAIN_NAME, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import sleep

from src.api.goldrush_gas import get_current_gas_snapshot


def append_once(output_path: Path) -> None:
    """Coleta um snapshot de gas price (GoldRush) e faz append em CSV."""
    df = get_current_gas_snapshot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False)


def run_loop() -> None:
    """
    Loop infinito para coleta contínua.

    Variáveis de ambiente:
    - DATA_PATH: caminho do CSV de saída (default: data/realtime/gas_price_history.csv)
    - COLLECT_INTERVAL_SECONDS: intervalo entre coletas (default: 60)
    """
    data_path = Path(os.getenv("DATA_PATH", "data/realtime/gas_price_history.csv"))
    interval = int(os.getenv("COLLECT_INTERVAL_SECONDS", "60"))

    while True:
        try:
            append_once(data_path)
            print(f"[collector] Linha gravada em {data_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[collector] Erro ao coletar gas price: {exc}")
        sleep(interval)


if __name__ == "__main__":
    mode = os.getenv("COLLECT_MODE", "loop")
    output = Path(os.getenv("DATA_PATH", "data/realtime/gas_price_history.csv"))

    if mode == "once":
        append_once(output)
    else:
        run_loop()
