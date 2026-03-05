"""CLI entry point: train all models, compare metrics, save all pipelines."""

import logging
import sys

from src.config import RAW_DATA_PATH
from src.model import save_model, train_and_evaluate
from src.preprocessing import load_and_clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not RAW_DATA_PATH.exists():
        logger.error("Data file not found: %s", RAW_DATA_PATH)
        logger.error("Run webscraping.py first to collect data.")
        sys.exit(1)

    df = load_and_clean(str(RAW_DATA_PATH))
    print(f"\nClean dataset: {len(df)} rows")
    print(f"Property types: {sorted(df['Type'].unique())}")
    print(f"Locations: {len(df['Location_Clean'].unique())} unique")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    result = train_and_evaluate(df)

    print(f"\n{'Model':<25} {'R2':>8} {'RMSE':>15} {'MAE':>15} {'CV R2 (mean+/-std)':>22}")
    print("-" * 87)
    for name, metrics in result["results"].items():
        marker = " (best)" if name == result["best_name"] else ""
        print(
            f"{name:<25} {metrics['r2']:>8.4f} {metrics['rmse']:>15,.0f} "
            f"{metrics['mae']:>15,.0f} {metrics['cv_r2_mean']:>8.4f}+/-{metrics['cv_r2_std']:.4f}{marker}"
        )

    best = result["best_name"]
    print(f"\nBest model: {best} (R2 = {result['results'][best]['r2']:.4f})")

    save_model(result["pipelines"], result["metadata"])
    print("All models saved to models/")


if __name__ == "__main__":
    main()
