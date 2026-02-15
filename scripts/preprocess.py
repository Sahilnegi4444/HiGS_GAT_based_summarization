"""
Preprocessing Pipeline — Train/Val/Test Splits

Usage:
    python scripts/preprocess.py \
        --input data/processed/newssumm_cleaned.parquet \
        --output data/processed

Generates:
    data/processed/train.parquet  (80%)
    data/processed/val.parquet    (10%)
    data/processed/test.parquet   (10%)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess NewsSumm dataset")
    parser.add_argument("--input", required=True, help="Path to cleaned parquet")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📖 Loading: {args.input}")
    df = pd.read_parquet(args.input)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"   Total: {len(df):,} samples")

    # Shuffle
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Split
    n = len(df)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))

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


if __name__ == "__main__":
    main()
