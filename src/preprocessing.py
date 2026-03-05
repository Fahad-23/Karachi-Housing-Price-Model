import logging

import numpy as np
import pandas as pd

from src.config import MIN_LOCATION_SAMPLES

logger = logging.getLogger(__name__)


def convert_price(price_str: str) -> float:
    """Convert price strings like '1.2 Crore' or '90 Lakh' to numeric PKR values."""
    try:
        price_str = str(price_str).lower().strip()
        price_str = price_str.replace("pkr", "").replace(",", "").strip()
        if "crore" in price_str:
            return float(price_str.replace("crore", "").strip()) * 1e7
        elif "lakh" in price_str:
            return float(price_str.replace("lakh", "").strip()) * 1e5
        else:
            return float(price_str)
    except (ValueError, TypeError) as e:
        logger.debug("Could not convert price '%s': %s", price_str, e)
        return np.nan


def convert_area(area_str: str) -> float:
    """Convert area string to square yards."""
    try:
        area_str = str(area_str).replace(",", "").lower().strip()
        if "sqft" in area_str:
            # 1 sq. yard = 9 sqft
            return float(area_str.replace("sqft", "").strip()) / 9
        elif "marla" in area_str:
            return float(area_str.replace("marla", "").strip()) * 30.25
        elif "kanal" in area_str:
            return float(area_str.replace("kanal", "").strip()) * 605
        else:
            return float(area_str.replace("sq. yd.", "").strip())
    except (ValueError, TypeError) as e:
        logger.debug("Could not convert area '%s': %s", area_str, e)
        return np.nan


def remove_outliers(df: pd.DataFrame, column: str = "Price") -> pd.DataFrame:
    """Remove outliers using IQR filtering (1.5x interquartile range)."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    logger.info("Outlier removal: %d -> %d rows (removed %d)", before, len(df), before - len(df))
    return df


def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sub-locality, falling back to parent for rare locations.

    Zameen.com format: "Sub-locality, Parent Locality"
    e.g. "Askari 5 - Sector J, Malir Cantonment"

    Using the sub-locality preserves granularity (Askari 5 vs Askari 6 have
    very different price points). Rare sub-localities (< MIN_LOCATION_SAMPLES)
    fall back to the parent to avoid one-hot encoding noise.
    """
    df["sub_locality"] = df["Location"].str.split(", ").str[0].str.replace('"', "")
    df["parent_locality"] = df["Location"].str.split(", ").str[-1].str.replace('"', "")

    # Count occurrences of each sub-locality
    sub_counts = df["sub_locality"].value_counts()
    rare_mask = df["sub_locality"].map(sub_counts) < MIN_LOCATION_SAMPLES

    # Use sub-locality when we have enough data, parent otherwise
    df["Location_Clean"] = df["sub_locality"]
    df.loc[rare_mask, "Location_Clean"] = df.loc[rare_mask, "parent_locality"]

    promoted = rare_mask.sum()
    logger.info(
        "Location cleaning: %d sub-localities, %d rare ones fell back to parent",
        df["sub_locality"].nunique(), promoted,
    )

    return df


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw CSV and return cleaned DataFrame ready for modeling."""
    df = pd.read_csv(filepath)
    logger.info("Loaded %d raw rows from %s", len(df), filepath)

    df["Price"] = df["Price"].apply(convert_price)
    df["Area"] = df["Area"].apply(convert_area)

    df = clean_location(df)

    df["Beds"] = pd.to_numeric(df["Beds"], errors="coerce")
    df["Baths"] = pd.to_numeric(df["Baths"], errors="coerce")

    required = ["Price", "Area", "Beds", "Baths", "Type", "Location_Clean"]
    df = df.dropna(subset=required)
    df = remove_outliers(df)

    # area_per_bed: captures property density (large houses with few bedrooms
    # are premium builds, small houses with many bedrooms are dense/cheap)
    df["area_per_bed"] = df["Area"] / df["Beds"].clip(lower=1)

    logger.info("Clean dataset: %d rows, %d unique locations",
                len(df), df["Location_Clean"].nunique())
    return df
