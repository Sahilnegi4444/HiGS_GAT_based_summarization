"""
Dataset Statistics & EDA

Usage:
    python scripts/compute_stats.py --data data/processed/newssumm_cleaned.parquet

Generates:
    data/processed/eda_distributions.png
    data/processed/eda_compression.png
    data/processed/eda_cluster_distribution.png
"""

import argparse
from collections import Counter

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set_palette("husl")


def compute_vocabulary_metrics(df, text_col="articles_clean"):
    """Compute total tokens, unique tokens, and type-token ratio."""
    all_tokens = []
    for text in df[text_col].dropna():
        all_tokens.extend(str(text).lower().split())
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0
    return total_tokens, unique_tokens, ttr


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--data", required=True, help="Path to cleaned parquet")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = data_path.parent

    print(f"📖 Loading: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   Total: {len(df):,} samples")

    # ── Token Statistics ──
    print(f"\n{'='*60}")
    print("📈 TOKEN STATISTICS")
    print(f"{'='*60}")

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
        print(f"\n{'='*60}")
        print("📂 CLUSTER (CATEGORY) STATISTICS")
        print(f"{'='*60}")
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
        print(f"\n{'='*60}")
        print("📚 VOCABULARY METRICS (Articles)")
        print(f"{'='*60}")
        total_tok, unique_tok, ttr = compute_vocabulary_metrics(df, "articles_clean")
        print(f"\n   Total tokens:           {total_tok:,}")
        print(f"   Unique tokens (vocab):  {unique_tok:,}")
        print(f"   Type-token ratio (TTR): {ttr:.6f}")

    if "summary_clean" in df.columns:
        print(f"\n{'='*60}")
        print("📚 VOCABULARY METRICS (Summaries)")
        print(f"{'='*60}")
        total_tok, unique_tok, ttr = compute_vocabulary_metrics(df, "summary_clean")
        print(f"\n   Total tokens:           {total_tok:,}")
        print(f"   Unique tokens (vocab):  {unique_tok:,}")
        print(f"   Type-token ratio (TTR): {ttr:.6f}")

    # ── Temporal Coverage ──
    if "year" in df.columns:
        print(f"\n{'='*60}")
        print("📅 TEMPORAL COVERAGE")
        print(f"{'='*60}")
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
    elif "date" in df.columns:
        print(f"\n{'='*60}")
        print("📅 TEMPORAL COVERAGE")
        print(f"{'='*60}")
        dates = pd.to_datetime(df["date"], errors="coerce")
        valid_years = dates.dt.year.dropna()
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
        print(f"\n{'='*60}")
        print("📰 SOURCE ANALYSIS")
        print(f"{'='*60}")
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

        # Use log scale for better visibility since cluster sizes vary widely
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

        # Add text annotation with summary stats
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

    print(f"\n✅ Done")


if __name__ == "__main__":
    main()
