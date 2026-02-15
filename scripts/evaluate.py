"""
Unified Evaluation Script — ROUGE + BERTScore

Usage:
    python scripts/evaluate.py \
        --predictions results/higs/predictions.csv \
        --output results/higs/evaluation.json

    Predictions CSV must have columns: 'generated', 'reference'
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization outputs")
    parser.add_argument("--predictions", required=True, help="CSV with generated/reference columns")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--bertscore-model", default="roberta-large",
                        help="Model for BERTScore (default: roberta-large)")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore computation")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load predictions
    print(f"📖 Loading: {args.predictions}")
    df = pd.read_csv(args.predictions)
    preds = df["generated"].fillna("").tolist()
    refs = df["reference"].fillna("").tolist()
    print(f"   Samples: {len(preds)}")

    results = {}

    # ROUGE
    print("\n📊 Computing ROUGE...")
    rouge = evaluate.load("rouge")
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    results["rouge1"] = round(r["rouge1"], 4)
    results["rouge2"] = round(r["rouge2"], 4)
    results["rougeL"] = round(r["rougeL"], 4)
    print(f"   ROUGE-1: {results['rouge1']}")
    print(f"   ROUGE-2: {results['rouge2']}")
    print(f"   ROUGE-L: {results['rougeL']}")

    # BERTScore
    if not args.no_bertscore:
        print(f"\n📊 Computing BERTScore (model: {args.bertscore_model})...")
        bertscore = evaluate.load("bertscore")
        b = bertscore.compute(
            predictions=preds,
            references=refs,
            lang="en",
            model_type=args.bertscore_model,
        )
        results["bertscore_precision"] = round(float(np.mean(b["precision"])), 4)
        results["bertscore_recall"] = round(float(np.mean(b["recall"])), 4)
        results["bertscore_f1"] = round(float(np.mean(b["f1"])), 4)
        print(f"   Precision: {results['bertscore_precision']}")
        print(f"   Recall:    {results['bertscore_recall']}")
        print(f"   F1:        {results['bertscore_f1']}")

    # Save
    results["num_samples"] = len(preds)
    results["bertscore_model"] = args.bertscore_model

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: {output_path}")

    # Print summary table
    print(f"\n{'='*40}")
    print(f"{'Metric':<20} {'Score':>10}")
    print(f"{'-'*40}")
    print(f"{'ROUGE-1':<20} {results['rouge1']:>10.4f}")
    print(f"{'ROUGE-2':<20} {results['rouge2']:>10.4f}")
    print(f"{'ROUGE-L':<20} {results['rougeL']:>10.4f}")
    if "bertscore_f1" in results:
        print(f"{'BERTScore F1':<20} {results['bertscore_f1']:>10.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
