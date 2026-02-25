import os
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
import requests

DateLike = Union[str, date, datetime]

API_KEY = os.getenv("API_KEY")


def _get_base_url() -> str:
    """
    Returns the Etherscan base URL configured via the BASE_URL environment variable.

    Expected value (for Ethereum mainnet v2 API):
        https://api.etherscan.io/v2/api
    """
    base_url = os.getenv("BASE_URL")
    if not base_url:
        raise RuntimeError("Environment variable 'BASE_URL' is not set")
    return base_url


def _normalise_date(value: DateLike) -> str:
    """Normalise a date-like value to the yyyy-MM-dd string format required by Etherscan."""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"Unsupported date type: {type(value)!r}")


def _call_etherscan(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Low-level helper to call the Etherscan v2 API.

    It injects the API key and a default chainid (1 = Ethereum mainnet),
    checks HTTP status codes and basic API error fields.
    """
    base_url = _get_base_url()

    api_key = API_KEY or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'API_KEY' is not set")

    merged_params = {"chainid": 1, "apikey": api_key, **params}

    response = requests.get(base_url, params=merged_params, timeout=10)
    response.raise_for_status()

    payload = response.json()

    # For most Etherscan endpoints, a non-"1" status indicates an error.
    status = payload.get("status")
    if status is not None and status != "1":
        message = payload.get("message", "Unknown error from Etherscan API")
        raise RuntimeError(f"Etherscan API error: {message}")

    return payload


def _ensure_dataframe(result: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from an iterable of dicts, returning an empty DataFrame if no data."""
    df = pd.DataFrame(list(result))
    if df.empty:
        return df
    return df


def _to_nullable_int(series: Any) -> pd.Series:
    """
    Convert a Series to a nullable Int64 column via numeric coercion.

    This wrapper evita problemas de tipagem estática (por exemplo, quando o
    analisador infere tipos como Timedelta) e garante que o resultado tenha
    o método ``astype`` disponível.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric).astype("Int64")


def get_daily_avg_gas_price(
    start_date: DateLike,
    end_date: DateLike,
    *,
    sort: str = "asc",
    chainid: int = 1,
) -> pd.DataFrame:
    """
    Retrieve daily average gas price statistics for a given period.

    Parameters
    ----------
    start_date, end_date:
        Boundaries of the analysis window. Can be datetime.date, datetime.datetime
        or strings in yyyy-MM-dd format.
    sort:
        "asc" (mais antigo primeiro) ou "desc" (mais recente primeiro).
    chainid:
        Identificador da chain (1 = Ethereum mainnet).

    Returns
    -------
    pd.DataFrame
        Colunas típicas:
        - UTCDate (datetime64[ns])
        - unixTimeStamp (int)
        - maxGasPrice_Wei, minGasPrice_Wei, avgGasPrice_Wei (float)
        - avgGasPrice_Gwei (float) – derivada para facilitar o uso em modelos.
    """
    start = _normalise_date(start_date)
    end = _normalise_date(end_date)

    payload = _call_etherscan(
        {
            "chainid": chainid,
            "module": "stats",
            "action": "dailyavggasprice",
            "startdate": start,
            "enddate": end,
            "sort": sort,
        }
    )

    df = _ensure_dataframe(payload.get("result", []))
    if df.empty:
        return df

    df["UTCDate"] = pd.to_datetime(df["UTCDate"], format="%Y-%m-%d", errors="coerce")

    for col in ("unixTimeStamp",):
        if col in df.columns:
            df[col] = _to_nullable_int(df[col])

    for col in ("maxGasPrice_Wei", "minGasPrice_Wei", "avgGasPrice_Wei"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "avgGasPrice_Wei" in df.columns:
        df["avgGasPrice_Gwei"] = df["avgGasPrice_Wei"] / 1e9

    df["date"] = df["UTCDate"]
    return df


def get_daily_avg_gas_limit(
    start_date: DateLike,
    end_date: DateLike,
    *,
    sort: str = "asc",
    chainid: int = 1,
) -> pd.DataFrame:
    """
    Retrieve daily average gas limit statistics for a given period.
    """
    start = _normalise_date(start_date)
    end = _normalise_date(end_date)

    payload = _call_etherscan(
        {
            "chainid": chainid,
            "module": "stats",
            "action": "dailyavggaslimit",
            "startdate": start,
            "enddate": end,
            "sort": sort,
        }
    )

    df = _ensure_dataframe(payload.get("result", []))
    if df.empty:
        return df

    df["UTCDate"] = pd.to_datetime(df["UTCDate"], format="%Y-%m-%d", errors="coerce")
    if "unixTimeStamp" in df.columns:
        df["unixTimeStamp"] = _to_nullable_int(df["unixTimeStamp"])

    if "gasLimit" in df.columns:
        df["gasLimit"] = pd.to_numeric(df["gasLimit"], errors="coerce")

    df["date"] = df["UTCDate"]
    return df


def get_daily_total_gas_used(
    start_date: DateLike,
    end_date: DateLike,
    *,
    sort: str = "asc",
    chainid: int = 1,
) -> pd.DataFrame:
    """
    Retrieve the total daily gas used in the network for a given period.
    """
    start = _normalise_date(start_date)
    end = _normalise_date(end_date)

    payload = _call_etherscan(
        {
            "chainid": chainid,
            "module": "stats",
            "action": "dailygasused",
            "startdate": start,
            "enddate": end,
            "sort": sort,
        }
    )

    df = _ensure_dataframe(payload.get("result", []))
    if df.empty:
        return df

    df["UTCDate"] = pd.to_datetime(df["UTCDate"], format="%Y-%m-%d", errors="coerce")
    if "unixTimeStamp" in df.columns:
        df["unixTimeStamp"] = _to_nullable_int(df["unixTimeStamp"])

    if "gasUsed" in df.columns:
        df["gasUsed"] = pd.to_numeric(df["gasUsed"], errors="coerce")

    df["date"] = df["UTCDate"]
    return df


def estimate_confirmation_time(
    gas_price_gwei: Union[int, float],
    *,
    chainid: int = 1,
) -> Optional[int]:
    """
    Estimate confirmation time (in seconds) for a given gas price.

    This uses the classic `gastracker` / `gasestimate` endpoint, where `gasprice`
    is specified in Gwei. If the endpoint returns no result, ``None`` is returned.
    """
    payload = _call_etherscan(
        {
            "chainid": chainid,
            "module": "gastracker",
            "action": "gasestimate",
            "gasprice": str(gas_price_gwei),
        }
    )

    result = payload.get("result")
    if result is None:
        return None

    try:
        return int(result)
    except (TypeError, ValueError):
        return None


def get_gas_data(start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """
    Backwards-compatible helper that delegates to ``get_daily_avg_gas_price``.

    This keeps o nome original usado no início do projeto, mas agora com
    parâmetros explícitos de período.
    """
    return get_daily_avg_gas_price(start_date=start_date, end_date=end_date)
