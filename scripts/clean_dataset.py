"""
Data Cleaning Pipeline for NewsSumm Dataset

Usage:
    python scripts/clean_dataset.py

Input:  data/NewsSumm_Dataset.xlsx
Output: data/processed/newssumm_cleaned.parquet
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_RAW / "NewsSumm_Dataset.xlsx"
OUTPUT_PATH = DATA_PROCESSED / "newssumm_cleaned.parquet"

# Column name standardization
COLUMN_MAPPING = {
    "newspaper_name": "source",
    "published_date": "date",
    "headline": "headline",
    "article_text": "articles",
    "human_summary": "summary",
    "news_category": "category",
}


def clean_html_tags(text: str) -> str:
    """Remove HTML tags and entities."""
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#[0-9]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    """Full text cleaning: HTML, URLs, emails, whitespace."""
    if pd.isna(text):
        return text
    text = str(text)
    text = clean_html_tags(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens."""
    if pd.isna(text) or text == "":
        return 0
    return len(str(text).split())


def main():
    print(f"📖 Loading dataset from: {DATASET_PATH}")
    df = pd.read_excel(DATASET_PATH)
    print(f"✅ Loaded: {len(df):,} rows × {df.shape[1]} columns")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\n", "", regex=False)
    existing = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=existing)
    print(f"✅ Mapped columns: {list(existing.values())}")

    # Apply text cleaning
    print("🧹 Cleaning text...")
    for col in ["articles", "summary", "headline"]:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].apply(clean_text)

    # Remove empty rows
    before = len(df)
    if "articles_clean" in df.columns:
        df = df[df["articles_clean"].str.len() > 100]
    if "summary_clean" in df.columns:
        df = df[df["summary_clean"].str.len() > 20]
    print(f"🗑️  Removed {before - len(df):,} rows with short/empty text")

    # Remove duplicates
    before = len(df)
    if "articles_clean" in df.columns:
        df = df.drop_duplicates(subset=["articles_clean"])
    print(f"🗑️  Removed {before - len(df):,} duplicates")

    # Token counts
    if "articles_clean" in df.columns:
        df["article_token_count"] = df["articles_clean"].apply(count_tokens)
    if "summary_clean" in df.columns:
        df["summary_token_count"] = df["summary_clean"].apply(count_tokens)
    if "article_token_count" in df.columns and "summary_token_count" in df.columns:
        df["compression_ratio"] = df.apply(
            lambda x: x["article_token_count"] / x["summary_token_count"]
            if x["summary_token_count"] > 0
            else np.nan,
            axis=1,
        )

    # Save
    cols = [
        c for c in [
            "source", "headline", "articles_clean", "summary_clean",
            "category", "article_token_count", "summary_token_count",
            "compression_ratio",
        ] if c in df.columns
    ]
    df_save = df[cols].copy()
    for col in df_save.select_dtypes(include=["object"]).columns:
        df_save[col] = df_save[col].astype(str)

    df_save.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved: {OUTPUT_PATH}")
    print(f"   Records: {len(df_save):,}")
    print(f"   Columns: {cols}")


if __name__ == "__main__":
    main()
