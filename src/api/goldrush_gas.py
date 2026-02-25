"""
Integração com a GoldRush API (Covalent) para obter gas prices em tempo real.

Endpoint usado:

    GET https://api.covalenthq.com/v1/{chainName}/event/{eventType}/gas_prices

As credenciais e parâmetros padrão vêm de variáveis de ambiente:

- GOLDRUSH_API_KEY        -> token usado no header Authorization: Bearer <token>
- GOLDRUSH_BASE_URL       -> base da API (default: https://api.covalenthq.com)
- GOLDRUSH_CHAIN_NAME     -> nome da chain, ex.: eth-mainnet
- GOLDRUSH_EVENT_TYPE     -> tipo de evento, ex.: erc20, uniswapv3, nativetokens
- GOLDRUSH_QUOTE_CURRENCY -> moeda de cotação (USD, BRL, etc.), opcional
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

GOLDRUSH_API_KEY = os.getenv("GOLDRUSH_API_KEY")
GOLDRUSH_BASE_URL = os.getenv("GOLDRUSH_BASE_URL", "https://api.covalenthq.com")


def _get_auth_header() -> Dict[str, str]:
    """Monta o header Authorization para chamadas à GoldRush."""
    if not GOLDRUSH_API_KEY:
        raise RuntimeError("Environment variable 'GOLDRUSH_API_KEY' is not set")
    return {"Authorization": f"Bearer {GOLDRUSH_API_KEY}"}


def get_gas_prices(
    chain_name: Optional[str] = None,
    event_type: Optional[str] = None,
    quote_currency: Optional[str] = None,
) -> pd.DataFrame:
    """
    Obtém gas prices em tempo real a partir da GoldRush API.

    Parameters
    ----------
    chain_name:
        Nome da chain (ex.: \"eth-mainnet\"). Se None, usa GOLDRUSH_CHAIN_NAME do .env.
    event_type:
        Tipo de evento (ex.: \"erc20\", \"uniswapv3\", \"nativetokens\").
        Se None, usa GOLDRUSH_EVENT_TYPE do .env (default: \"erc20\").
    quote_currency:
        Moeda de cotação (ex.: \"USD\"). Se None, usa GOLDRUSH_QUOTE_CURRENCY do .env.

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha por faixa de gas (items), contendo também
        metadados da resposta (chain_id, chain_name, updated_at, etc.).
    """
    chain_name = chain_name or os.getenv("GOLDRUSH_CHAIN_NAME", "eth-mainnet")
    event_type = event_type or os.getenv("GOLDRUSH_EVENT_TYPE", "erc20")
    quote_currency = quote_currency or os.getenv("GOLDRUSH_QUOTE_CURRENCY", "USD")

    url = f"{GOLDRUSH_BASE_URL}/v1/{chain_name}/event/{event_type}/gas_prices"

    headers = _get_auth_header()
    params: Dict[str, Any] = {
        "quote-currency": quote_currency,
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)

    if response.status_code == 401:
        raise RuntimeError(
            "GoldRush API returned 401 Unauthorized. "
            "Verifique se GOLDRUSH_API_KEY está correta e ativa."
        )

    response.raise_for_status()
    payload = response.json()

    # A resposta da GoldRush vem aninhada em "data".
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Resposta inesperada da GoldRush API (campo 'data' ausente).")

    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        # Ainda assim retornamos um DF vazio para evitar quebra do fluxo.
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # Anexa metadados em colunas para cada linha.
    meta_fields = [
        "chain_id",
        "chain_name",
        "quote_currency",
        "updated_at",
        "event_type",
        "gas_quote_rate",
        "base_fee",
    ]
    for field in meta_fields:
        if field in data:
            df[field] = data[field]

    # Converte updated_at para datetime quando possível.
    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    # Marca o momento da coleta local como timestamp_utc.
    df["timestamp_utc"] = datetime.utcnow()

    return df


def get_current_gas_snapshot(
    chain_name: Optional[str] = None,
    event_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retorna uma única linha com snapshot atual de gas (para coleta contínua/append em CSV).

    Colunas: timestamp_utc, base_fee, chain_name, updated_at e demais metadados
    da resposta. Útil para collect_realtime_gas e pipelines que anexam leituras em CSV.
    """
    df = get_gas_prices(chain_name=chain_name, event_type=event_type)
    if df.empty:
        return pd.DataFrame({
            "timestamp_utc": [],
            "base_fee": [],
            "chain_name": [],
            "updated_at": [],
        })
    # Uma linha com o estado atual (primeira faixa + metadados)
    row = df.iloc[0:1].copy()
    return row


__all__ = ["get_gas_prices", "get_current_gas_snapshot"]

