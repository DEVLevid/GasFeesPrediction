"""
Pipeline de coleta de dados de gas a partir da API Etherscan.

Responsável por:
- Orquestrar chamadas para os endpoints diários de gas.
- Consolidar os resultados em um único DataFrame indexado por data.
- Salvar datasets em disco para uso posterior em modelagem.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.api.getGasFeeData import (
    DateLike,
    get_daily_avg_gas_limit,
    get_daily_avg_gas_price,
    get_daily_total_gas_used,
)


@dataclass
class GasDataConfig:
    """Configuração básica para coleta de dados de gas."""

    start_date: DateLike
    end_date: DateLike
    chainid: int = 1
    sort: str = "asc"


def fetch_historical_gas_data(config: GasDataConfig) -> pd.DataFrame:
    """
    Coleta dados históricos de gas (preço médio, limite médio e gas total usado)
    para o período configurado e consolida em um único DataFrame.

    O merge é feito pela coluna ``date`` (derivada de ``UTCDate`` em cada endpoint).
    """
    price_df = get_daily_avg_gas_price(
        config.start_date,
        config.end_date,
        sort=config.sort,
        chainid=config.chainid,
    )
    limit_df = get_daily_avg_gas_limit(
        config.start_date,
        config.end_date,
        sort=config.sort,
        chainid=config.chainid,
    )
    used_df = get_daily_total_gas_used(
        config.start_date,
        config.end_date,
        sort=config.sort,
        chainid=config.chainid,
    )

    dfs = [df for df in (price_df, limit_df, used_df) if not df.empty]
    if not dfs:
        return pd.DataFrame()

    # Começa com o primeiro DataFrame e faz merges sucessivos por data.
    combined = dfs[0].copy()
    for extra_df in dfs[1:]:
        combined = combined.merge(
            extra_df,
            on="date",
            how="outer",
            suffixes=("", "_dup"),
        )

    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def save_gas_data(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """
    Salva o DataFrame de gas em CSV no caminho especificado.

    Retorna o caminho final salvo.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _default_period(days: int = 365) -> tuple[date, date]:
    """Retorna um período padrão (hoje menos N dias até hoje)."""
    end = date.today()
    start = end - timedelta(days=days)
    return start, end


def run_cli(
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = 365,
    output_dir: Union[str, Path] = "data/raw",
) -> Path:
    """
    Ponto de entrada programático para coleta via linha de comando.

    Se ``start`` e ``end`` forem fornecidos (formato yyyy-MM-dd), eles são usados.
    Caso contrário, utiliza o período padrão de ``days`` dias até hoje.
    """
    if start and end:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    else:
        start_date, end_date = _default_period(days=days)

    config = GasDataConfig(start_date=start_date, end_date=end_date)
    df = fetch_historical_gas_data(config)

    if df.empty:
        raise RuntimeError("Nenhum dado de gas retornado para o período especificado.")

    output_dir_path = Path(output_dir)
    filename = f"gas_data_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"
    output_path = output_dir_path / filename
    return save_gas_data(df, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Coleta dados diários de gas a partir da API Etherscan."
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Data inicial no formato yyyy-MM-dd (opcional se --days for usado).",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Data final no formato yyyy-MM-dd (opcional se --days for usado).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Quantidade de dias até hoje para o período padrão (default: 365).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Diretório de saída para o CSV gerado (default: data/raw).",
    )

    args = parser.parse_args()

    path = run_cli(
        start=args.start,
        end=args.end,
        days=args.days,
        output_dir=args.output_dir,
    )
    print(f"Dados de gas salvos em: {path}")

