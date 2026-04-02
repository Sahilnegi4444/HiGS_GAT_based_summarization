"""
Data Preparation Pipeline — Clean, Split & Statistics
======================================================

Combines dataset cleaning, train/val/test splitting, and basic statistics
into a single pipeline.

Usage:
    # Run full pipeline (clean → split → stats)
    python scripts/prepare_data.py

    # Run specific steps
    python scripts/prepare_data.py --step clean
    python scripts/prepare_data.py --step split
    python scripts/prepare_data.py --step stats
    python scripts/prepare_data.py --step all    # (default)

Input:  data/NewsSumm_Dataset.xlsx
Output: data/processed/newssumm_cleaned.parquet
        data/processed/train.parquet  (80%)
        data/processed/val.parquet    (10%)
        data/processed/test.parquet   (10%)
        data/processed/eda_distributions.png
        data/processed/eda_compression.png
        data/processed/eda_cluster_distribution.png
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATASET_PATH = DATA_RAW / "NewsSumm_Dataset.xlsx"
CLEANED_PATH = DATA_PROCESSED / "newssumm_cleaned.parquet"

# Column name standardization
COLUMN_MAPPING = {
    "newspaper_name": "source",
    "published_date": "date",
    "headline": "headline",
    "article_text": "articles",
    "human_summary": "summary",
    "news_category": "category",
}


# ═══════════════════════════════════════════════════════════════════
# STEP 1: CLEAN
# ═══════════════════════════════════════════════════════════════════

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


def step_clean(input_path=None, output_path=None):
    """Clean raw dataset: HTML stripping, URL removal, deduplication, token counting."""
    input_path = input_path or DATASET_PATH
    output_path = output_path or CLEANED_PATH

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("STEP 1: CLEAN DATASET")
    print(f"{'='*60}")
    print(f"📖 Loading dataset from: {input_path}")
    df = pd.read_excel(input_path)
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

    # Parse date column
    if "date" in df.columns:
        print("📅 Parsing date column...")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year.astype("Int64")

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
            if x["summary_token_count"] > 0 else np.nan,
            axis=1,
        )

    # Save
    cols = [
        c for c in [
            "source", "date", "year", "headline", "articles_clean",
            "summary_clean", "category", "article_token_count",
            "summary_token_count", "compression_ratio",
        ] if c in df.columns
    ]
    df_save = df[cols].copy()
    for col in df_save.select_dtypes(include=["object"]).columns:
        df_save[col] = df_save[col].astype(str)

    df_save.to_parquet(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    print(f"   Records: {len(df_save):,}")
    print(f"   Columns: {cols}")

    return output_path


# ═══════════════════════════════════════════════════════════════════
# STEP 2: SPLIT
# ═══════════════════════════════════════════════════════════════════

def step_split(input_path=None, output_dir=None, train_ratio=0.8,
               val_ratio=0.1, seed=42, max_samples=None):
    """Create 80/10/10 train/val/test splits from cleaned dataset."""
    input_path = input_path or CLEANED_PATH
    output_dir = Path(output_dir) if output_dir else DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("STEP 2: CREATE TRAIN/VAL/TEST SPLITS")
    print(f"{'='*60}")
    print(f"📖 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    if max_samples:
        df = df.head(max_samples)
    print(f"   Total: {len(df):,} samples")

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Save
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"\n✅ Splits saved to {output_dir}/")
    print(f"   Train: {len(train_df):,} ({len(train_df)/n:.0%})")
    print(f"   Val:   {len(val_df):,} ({len(val_df)/n:.0%})")
    print(f"   Test:  {len(test_df):,} ({len(test_df)/n:.0%})")

    # Summary statistics
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if "article_token_count" in split.columns:
            print(f"\n   {name}:")
            print(f"     Avg article tokens: {split['article_token_count'].mean():.0f}")
            print(f"     Avg summary tokens: {split['summary_token_count'].mean():.0f}")


# ═══════════════════════════════════════════════════════════════════
# STEP 3: STATISTICS & EDA
# ═══════════════════════════════════════════════════════════════════

def compute_vocabulary_metrics(df, text_col="articles_clean"):
    """Compute total tokens, unique tokens, and type-token ratio."""
    all_tokens = []
    for text in df[text_col].dropna():
        all_tokens.extend(str(text).lower().split())
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0
    return total_tokens, unique_tokens, ttr


def step_stats(input_path=None):
    """Compute and display dataset statistics with EDA plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("ggplot")
    sns.set_palette("husl")

    input_path = Path(input_path) if input_path else CLEANED_PATH
    output_dir = input_path.parent

    print(f"\n{'='*60}")
    print("STEP 3: DATASET STATISTICS & EDA")
    print(f"{'='*60}")
    print(f"📖 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Total: {len(df):,} samples")

    # ── Token Statistics ──
    print(f"\n{'─'*40}")
    print("📈 TOKEN STATISTICS")
    print(f"{'─'*40}")

    if "article_token_count" in df.columns:
        print(f"\n📰 Articles:")
        print(f"   Mean:   {df['article_token_count'].mean():,.1f}")
        print(f"   Median: {df['article_token_count'].median():,.1f}")
        print(f"   Min:    {df['article_token_count'].min():,}")
        print(f"   Max:    {df['article_token_count'].max():,}")
        print(f"   Std:    {df['article_token_count'].std():,.1f}")

    if "summary_token_count" in df.columns:
        print(f"\n📝 Summaries:")
        print(f"   Mean:   {df['summary_token_count'].mean():,.1f}")
        print(f"   Median: {df['summary_token_count'].median():,.1f}")
        print(f"   Min:    {df['summary_token_count'].min():,}")
        print(f"   Max:    {df['summary_token_count'].max():,}")
        print(f"   Std:    {df['summary_token_count'].std():,.1f}")

    if "compression_ratio" in df.columns:
        valid = df["compression_ratio"].dropna()
        print(f"\n🔄 Compression Ratio: {valid.mean():.2f}x (mean)")

    # ── Cluster (Category) Statistics ──
    if "category" in df.columns:
        print(f"\n{'─'*40}")
        print("📂 CLUSTER (CATEGORY) STATISTICS")
        print(f"{'─'*40}")
        category_counts = df["category"].value_counts()
        n_categories = len(category_counts)
        avg_docs = category_counts.mean()
        min_docs = category_counts.min()
        max_docs = category_counts.max()
        min_cat = category_counts.idxmin()
        max_cat = category_counts.idxmax()
        print(f"\n   Total unique categories (clusters): {n_categories:,}")
        print(f"   Average documents per cluster:      {avg_docs:,.1f}")
        print(f"   Minimum documents per cluster:      {min_docs:,} ('{min_cat}')")
        print(f"   Maximum documents per cluster:      {max_docs:,} ('{max_cat}')")
        print(f"\n   Top 10 Categories:")
        for cat, cnt in category_counts.head(10).items():
            print(f"      {cat}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # ── Vocabulary Metrics ──
    if "articles_clean" in df.columns:
        print(f"\n{'─'*40}")
        print("📚 VOCABULARY METRICS (Articles)")
        print(f"{'─'*40}")
        total_tok, unique_tok, ttr = compute_vocabulary_metrics(df, "articles_clean")
        print(f"\n   Total tokens:           {total_tok:,}")
        print(f"   Unique tokens (vocab):  {unique_tok:,}")
        print(f"   Type-token ratio (TTR): {ttr:.6f}")

    if "summary_clean" in df.columns:
        print(f"\n{'─'*40}")
        print("📚 VOCABULARY METRICS (Summaries)")
        print(f"{'─'*40}")
        total_tok, unique_tok, ttr = compute_vocabulary_metrics(df, "summary_clean")
        print(f"\n   Total tokens:           {total_tok:,}")
        print(f"   Unique tokens (vocab):  {unique_tok:,}")
        print(f"   Type-token ratio (TTR): {ttr:.6f}")

    # ── Temporal Coverage ──
    if "year" in df.columns:
        print(f"\n{'─'*40}")
        print("📅 TEMPORAL COVERAGE")
        print(f"{'─'*40}")
        valid_years = df["year"].dropna()
        if len(valid_years) > 0:
            earliest = int(valid_years.min())
            latest = int(valid_years.max())
            span = latest - earliest
            print(f"\n   Earliest year:  {earliest}")
            print(f"   Latest year:    {latest}")
            print(f"   Total span:     {span} years")
            print(f"   Articles with valid dates: {len(valid_years):,} / {len(df):,}")
        else:
            print("\n   ⚠️  No valid dates found in the dataset")

    # ── Source Analysis ──
    if "source" in df.columns:
        print(f"\n{'─'*40}")
        print("📰 SOURCE ANALYSIS")
        print(f"{'─'*40}")
        source_counts = df["source"].value_counts()
        n_sources = len(source_counts)
        avg_articles = source_counts.mean()
        print(f"\n   Total unique sources: {n_sources:,}")
        print(f"   Average articles per source: {avg_articles:,.1f}")
        print(f"\n   Top 10 Sources:")
        for src, cnt in source_counts.head(10).items():
            print(f"      {src}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # ── Visualizations ──
    print(f"\n📊 Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NewsSumm Dataset — Distribution Analysis",
                 fontsize=16, fontweight="bold", y=1.02)

    if "article_token_count" in df.columns:
        ax = axes[0, 0]
        df["article_token_count"].hist(bins=50, ax=ax, color="steelblue",
                                       edgecolor="white", alpha=0.8)
        ax.axvline(df["article_token_count"].mean(), color="red", linestyle="--",
                   label=f"Mean: {df['article_token_count'].mean():.0f}")
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Frequency")
        ax.set_title("Article Length Distribution")
        ax.legend()
        ax.set_xlim(0, df["article_token_count"].quantile(0.99))

    if "summary_token_count" in df.columns:
        ax = axes[0, 1]
        df["summary_token_count"].hist(bins=50, ax=ax, color="seagreen",
                                       edgecolor="white", alpha=0.8)
        ax.axvline(df["summary_token_count"].mean(), color="red", linestyle="--",
                   label=f"Mean: {df['summary_token_count'].mean():.0f}")
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Frequency")
        ax.set_title("Summary Length Distribution")
        ax.legend()

    if "category" in df.columns:
        ax = axes[1, 0]
        cats = df["category"].value_counts().head(10)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))
        cats.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Categories")
        ax.invert_yaxis()

    if "source" in df.columns:
        ax = axes[1, 1]
        srcs = df["source"].value_counts().head(8)
        colors = plt.cm.Set2(np.linspace(0, 1, len(srcs)))
        ax.pie(srcs.values, labels=srcs.index, autopct="%1.1f%%",
               colors=colors, explode=[0.02] * len(srcs))
        ax.set_title("Articles by Source")

    plt.tight_layout()
    path1 = output_dir / "eda_distributions.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"   ✓ {path1}")

    # Compression plot
    if "article_token_count" in df.columns and "summary_token_count" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sample = df.sample(n=min(10000, len(df)), random_state=42)
        axes[0].scatter(sample["article_token_count"], sample["summary_token_count"],
                        alpha=0.3, s=10, c="steelblue")
        axes[0].set_xlabel("Article Tokens")
        axes[0].set_ylabel("Summary Tokens")
        axes[0].set_title("Article vs Summary Length")

        valid = df["compression_ratio"].dropna()
        valid = valid[valid < valid.quantile(0.99)]
        valid.hist(bins=50, ax=axes[1], color="coral", edgecolor="white", alpha=0.8)
        axes[1].axvline(valid.mean(), color="red", linestyle="--",
                        label=f"Mean: {valid.mean():.1f}x")
        axes[1].set_xlabel("Compression Ratio")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Compression Ratio Distribution")
        axes[1].legend()

        plt.tight_layout()
        path2 = output_dir / "eda_compression.png"
        plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"   ✓ {path2}")

    # Cluster distribution histogram
    if "category" in df.columns:
        category_counts = df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(category_counts.values, bins=50, color="darkorange",
                edgecolor="white", alpha=0.85)
        ax.axvline(category_counts.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {category_counts.mean():.0f}")
        ax.axvline(category_counts.median(), color="blue", linestyle="-.", linewidth=2,
                   label=f"Median: {category_counts.median():.0f}")
        ax.set_xlabel("Number of Documents per Cluster", fontsize=13)
        ax.set_ylabel("Number of Clusters", fontsize=13)
        ax.set_title("Distribution of Cluster (Category) Sizes",
                     fontsize=15, fontweight="bold")
        ax.legend(fontsize=12)
        stats_text = (
            f"Total clusters: {len(category_counts):,}\n"
            f"Avg docs/cluster: {category_counts.mean():.1f}\n"
            f"Min docs/cluster: {category_counts.min():,}\n"
            f"Max docs/cluster: {category_counts.max():,}"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))
        plt.tight_layout()
        path3 = output_dir / "eda_cluster_distribution.png"
        plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"   ✓ {path3}")

    print(f"\n✅ Statistics complete")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data Preparation Pipeline — Clean, Split & Statistics"
    )
    parser.add_argument("--step", default="all",
                        choices=["clean", "split", "stats", "all"],
                        help="Which step to run (default: all)")
    parser.add_argument("--input", default=None,
                        help="Input file path (Excel for clean, Parquet for split/stats)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/processed/)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.step in ("clean", "all"):
        cleaned_path = step_clean(
            input_path=args.input if args.step == "clean" else None,
            output_path=Path(args.output_dir) / "newssumm_cleaned.parquet"
                        if args.output_dir else None,
        )

    if args.step in ("split", "all"):
        step_split(
            input_path=args.input if args.step == "split" else None,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_samples=args.max_samples,
        )

    if args.step in ("stats", "all"):
        step_stats(
            input_path=args.input if args.step == "stats" else None,
        )

    if args.step == "all":
        print(f"\n{'='*60}")
        print("✅ FULL PIPELINE COMPLETE")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
