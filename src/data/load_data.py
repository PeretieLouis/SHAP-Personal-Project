"""Load the Adult Income dataset from local raw files.

Combines the UCI train and test splits into a single DataFrame,
cleans formatting quirks, and returns a tidy DataFrame ready for
preprocessing.
"""

import pandas as pd

from src.config import DATA_RAW_DIR, TARGET_COLUMN

# Column names defined in adult.names
_COLUMN_NAMES: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    TARGET_COLUMN,
]


def load_raw_data() -> pd.DataFrame:
    """Load and combine the Adult Income train/test CSVs.

    Returns
    -------
    pd.DataFrame
        Combined dataset with cleaned column names and labels.
    """
    train_path = DATA_RAW_DIR / "adult.data"
    test_path = DATA_RAW_DIR / "adult.test"

    # Load training data
    df_train = pd.read_csv(
        train_path,
        names=_COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skipinitialspace=True,
    )

    # Load test data — first line is a comment "|1x3 Cross validator"
    df_test = pd.read_csv(
        test_path,
        names=_COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )

    # Test labels have a trailing dot (e.g. "<=50K.") — remove it
    df_test[TARGET_COLUMN] = df_test[TARGET_COLUMN].str.rstrip(".")

    # Combine into a single dataset
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    return df


def load_data_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict of the loaded data.

    Parameters
    ----------
    df : pd.DataFrame
        The raw combined dataset.

    Returns
    -------
    dict
        Keys: n_rows, n_cols, n_missing, target_distribution.
    """
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_missing": int(df.isna().sum().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
    }
