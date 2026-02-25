"""
Mineração de dados históricos por bloco via GoldRush (Covalent) block_v2.

Gera um dataset em CSV para treino de modelo de predição de Gas Fees (TCC).
Endpoint: GET /v1/{chainName}/block_v2/{blockHeight}/

Variáveis de ambiente (carregadas do .env):
- GOLDRUSH_API_KEY   -> chave para Authorization: Bearer <token>
- GOLDRUSH_BASE_URL  -> base da API (default: https://api.covalenthq.com)
- GOLDRUSH_CHAIN_NAME -> chain, ex.: eth-mainnet ou matic-mainnet
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

# Chave da API: pode ser definida no .env (GOLDRUSH_API_KEY) para o TCC
API_KEY: Optional[str] = os.getenv("GOLDRUSH_API_KEY")
GOLDRUSH_BASE_URL = os.getenv("GOLDRUSH_BASE_URL", "https://api.covalenthq.com")

# Constantes de resiliência: intervalo entre requisições e backoff em rate limit
REQUEST_INTERVAL_SEC = 0.3
RATE_LIMIT_BACKOFF_SEC = 10
MAX_RETRIES = 3

# Referência para converter data → bloco (aproximado): bloco conhecido e data UTC
# Ethereum ~7200 blocos/dia; Polygon (matic) ~43200 blocos/dia
_REF_BLOCK_ETH = 18_900_000
_REF_DATE_ETH = datetime(2023, 12, 30, 18, 5, 35)  # UTC, conforme block_v2
_BLOCKS_PER_DAY: dict[str, float] = {"eth-mainnet": 7200.0, "matic-mainnet": 43200.0}


def _date_to_block_range(
    start_date: str,
    end_date: str,
    chain_name: str,
) -> tuple[int, int]:
    """
    Converte um intervalo de datas (YYYY-MM-DD) em intervalo de blocos aproximado.
    Usa referência fixa (bloco conhecido em uma data) e blocos/dia por chain.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start > end:
        raise ValueError("start_date deve ser <= end_date")
    blocks_per_day = _BLOCKS_PER_DAY.get(chain_name, 7200.0)
    ref_midnight = _REF_DATE_ETH.replace(hour=0, minute=0, second=0, microsecond=0)
    start_days = (start - ref_midnight).days
    end_days = (end - ref_midnight).days
    start_block = max(0, int(_REF_BLOCK_ETH + start_days * blocks_per_day))
    end_block = max(start_block, int(_REF_BLOCK_ETH + (end_days + 1) * blocks_per_day))
    return start_block, end_block


def _auth_headers(api_key: str) -> dict[str, str]:
    """Monta o header Authorization para chamadas à GoldRush."""
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _get_block_v2(
    base_url: str,
    chain_name: str,
    block_height: int,
    api_key: str,
) -> Optional[dict[str, Any]]:
    """
    Requisição GET para block_v2. Retorna o payload JSON ou None em falha.
    Em 429 (rate limit) faz backoff e retenta até MAX_RETRIES.
    """
    url = f"{base_url.rstrip('/')}/v1/{chain_name}/block_v2/{block_height}/"
    headers = _auth_headers(api_key)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(RATE_LIMIT_BACKOFF_SEC)
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                return None
            return body
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RATE_LIMIT_BACKOFF_SEC)
    return None


def _get_tx_count_from_link(transactions_link: str, api_key: str) -> Optional[int]:
    """
    Obtém a quantidade de transações do bloco via transactions_v3.
    Usa o link retornado em block_v2 (block_hash/transactions_v3/).
    Em caso de paginação, retorna o total da primeira página (subestimativa).
    """
    headers = _auth_headers(api_key)
    try:
        resp = requests.get(transactions_link, headers=headers, timeout=15)
        if resp.status_code == 429:
            time.sleep(RATE_LIMIT_BACKOFF_SEC)
            resp = requests.get(transactions_link, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return None
        items = data.get("data", {}).get("items") or []
        pagination = (data.get("data") or {}).get("pagination") or {}
        total = pagination.get("total_count")
        if total is not None:
            return int(total)
        return len(items)
    except requests.RequestException:
        return None


def fetch_blockchain_data(
    start_block: int,
    end_block: int,
    api_key: Optional[str] = None,
    chain_name: Optional[str] = None,
    include_tx_count: bool = False,
) -> pd.DataFrame:
    """
    Coleta dados de blocos no range [start_block, end_block] via GoldRush block_v2.

    Campos por bloco:
    - block_height: altura do bloco (height).
    - signed_at: timestamp ISO em que o bloco foi assinado.
    - gas_used: gás total consumido no bloco.
    - gas_limit: limite de gás do bloco.
    - base_fee: base fee (EIP-1559), se disponível na API; caso contrário NaN.
    - tx_count: número de transações (apenas se include_tx_count=True; senão NaN).

    Em erros de conexão ou rate limit (429): backoff e retentativas; se falhar,
    a linha do bloco é preenchida com NaN e o loop continua (resiliência).
    """
    key = api_key or API_KEY
    if not key:
        raise RuntimeError(
            "API key não definida. Configure GOLDRUSH_API_KEY no .env ou passe api_key."
        )
    chain = chain_name or os.getenv("GOLDRUSH_CHAIN_NAME", "eth-mainnet")
    base_url = GOLDRUSH_BASE_URL.rstrip("/")

    rows: list[dict[str, Any]] = []
    for height in tqdm(
        range(start_block, end_block + 1),
        desc="Blocos",
        unit="bloco",
    ):
        payload = _get_block_v2(base_url, chain, height, key)
        if not payload:
            rows.append({
                "block_height": height,
                "signed_at": pd.NA,
                "gas_used": pd.NA,
                "gas_limit": pd.NA,
                "base_fee": pd.NA,
                "tx_count": pd.NA if include_tx_count else None,
            })
            time.sleep(REQUEST_INTERVAL_SEC)
            continue

        data = payload.get("data") or {}
        items = data.get("items") or []
        if not items:
            rows.append({
                "block_height": height,
                "signed_at": pd.NA,
                "gas_used": pd.NA,
                "gas_limit": pd.NA,
                "base_fee": pd.NA,
                "tx_count": pd.NA if include_tx_count else None,
            })
            time.sleep(REQUEST_INTERVAL_SEC)
            continue

        block = items[0]
        # block_height (altura do bloco)
        block_height = block.get("height", height)
        # signed_at (timestamp ISO)
        signed_at = block.get("signed_at")
        # gas_used (gás total consumido)
        gas_used = block.get("gas_used")
        # gas_limit (limite de gás do bloco)
        gas_limit = block.get("gas_limit")
        # base_fee (EIP-1559; API pode não retornar)
        base_fee = block.get("base_fee", pd.NA)

        tx_count: Any = None
        if include_tx_count:
            link = block.get("transactions_link")
            tx_count = _get_tx_count_from_link(link, key) if link else None
            if tx_count is None:
                tx_count = pd.NA
            time.sleep(REQUEST_INTERVAL_SEC)

        rows.append({
            "block_height": block_height,
            "signed_at": signed_at,
            "gas_used": gas_used,
            "gas_limit": gas_limit,
            "base_fee": base_fee,
            "tx_count": tx_count,
        })
        time.sleep(REQUEST_INTERVAL_SEC)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minera dados históricos de blocos (GoldRush block_v2). Retroativo: por blocos ou por data."
    )
    parser.add_argument(
        "start_block",
        type=int,
        nargs="?",
        default=None,
        help="Bloco inicial (inclusive). Obrigatório se não usar --start-date/--end-date.",
    )
    parser.add_argument(
        "end_block",
        type=int,
        nargs="?",
        default=None,
        help="Bloco final (inclusive). Obrigatório se não usar --start-date/--end-date.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        metavar="YYYY-MM-DD",
        help="Data inicial para coleta retroativa (use com --end-date em vez de blocos).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        metavar="YYYY-MM-DD",
        help="Data final para coleta retroativa (use com --start-date).",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=os.getenv("GOLDRUSH_CHAIN_NAME", "eth-mainnet"),
        help="Chain (ex.: eth-mainnet, matic-mainnet). Default: GOLDRUSH_CHAIN_NAME ou eth-mainnet",
    )
    parser.add_argument(
        "--tx-count",
        action="store_true",
        help="Incluir número de transações por bloco (requisição extra por bloco)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/gas_data_tcc.csv"),
        help="Caminho do CSV de saída. Default: data/gas_data_tcc.csv",
    )
    args = parser.parse_args()

    start_block = args.start_block
    end_block = args.end_block
    if args.start_date and args.end_date:
        start_block, end_block = _date_to_block_range(
            args.start_date, args.end_date, args.chain
        )
        print(f"Intervalo {args.start_date} a {args.end_date} → blocos {start_block} a {end_block}")
    if start_block is None or end_block is None:
        parser.error("Informe start_block e end_block, ou --start-date e --end-date")
    if start_block > end_block:
        parser.error("start_block deve ser <= end_block")

    df = fetch_blockchain_data(
        start_block,
        end_block,
        api_key=API_KEY,
        chain_name=args.chain,
        include_tx_count=args.tx_count,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Dataset salvo: {args.output.resolve()} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
