"""
Dataset Statistics & EDA

Usage:
    python scripts/compute_stats.py --data data/processed/newssumm_cleaned.parquet

Generates:
    data/processed/eda_distributions.png
    data/processed/eda_compression.png
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set_palette("husl")


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

    if "category" in df.columns:
        print(f"\n📂 Categories: {df['category'].nunique()} unique")
        for cat, cnt in df["category"].value_counts().head(10).items():
            print(f"   {cat}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    if "source" in df.columns:
        print(f"\n📰 Sources: {df['source'].nunique()} unique")
        for src, cnt in df["source"].value_counts().items():
            print(f"   {src}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

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

    print(f"\n✅ Done")


if __name__ == "__main__":
    main()
