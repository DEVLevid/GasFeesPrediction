from pathlib import Path

from src.api.goldrush_gas import get_gas_prices


OUTPUT_PATH = Path("data/goldrush_gas_sample.csv")


def main() -> None:
    # Chama a API da GoldRush usando as configs do .env
    df = get_gas_prices()

    # Mostra um resumo no terminal
    print("Primeiras linhas do DataFrame retornado:")
    print(df.head())

    # Salva em CSV para inspecionar depois
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAmostra de dados salva em: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()