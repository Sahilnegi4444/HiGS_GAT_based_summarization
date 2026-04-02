"""
HiGS Evaluation Script — Scores, Factual Consistency & Inference Costs
=======================================================================

Evaluates the HiGS model on the test split and reports:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - SacreBLEU
  - BERTScore (F1)
  - Entity-Based Factual Consistency (Precision, Hallucination Rate)
  - Inference Costs (Latency, Throughput, Peak VRAM)

Usage:
    python scripts/evaluate_higs.py
    python scripts/evaluate_higs.py --model data/HiGS/higs_model.pt --samples 100
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import spacy
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BartTokenizer

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.higs_model import HiGraphSum, split_into_sentences, extract_entities

# ── CHECKPOINT KEY REMAPPING ────────────────────────────────────────
def remap_checkpoint_keys(state_dict):
    """Remap old checkpoint keys to the Dual-Encoder architecture."""
    new_sd = {}
    for key, value in state_dict.items():
        if key == "projection.weight":
            new_sd["projection.0.weight"] = value
        elif key == "projection.bias":
            new_sd["projection.0.bias"] = value
        elif key in ("residual_gate",) or "word_projection" in key:
            continue
        else:
            new_sd[key] = value
    return new_sd


def main():
    import argparse
    import evaluate as hf_evaluate

    parser = argparse.ArgumentParser(description="Evaluate HiGS model")
    parser.add_argument("--model", default="data/HiGS/higs_model.pt",
                        help="Path to model checkpoint (default: data/HiGS/higs_model.pt)")
    parser.add_argument("--data", default="data/newssumm_cleaned.parquet",
                        help="Path to cleaned dataset")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of test samples to evaluate")
    parser.add_argument("--output", default="results/evaluation_report.json",
                        help="Output JSON path for results")
    parser.add_argument("--max-sents", type=int, default=30)
    parser.add_argument("--max-sent-len", type=int, default=64)
    parser.add_argument("--max-summary-len", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Load Model ──
    print(f"\n📦 Loading HiGS model from: {args.model}")
    model = HiGraphSum().to(device)

    if os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = remap_checkpoint_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        epoch = ckpt.get("epoch", "?")
        train_loss = ckpt.get("train_loss", "?")
        val_loss = ckpt.get("val_loss", "?")
        print(f"  ✅ Loaded (epoch={epoch}, train_loss={train_loss}, val_loss={val_loss})")
    else:
        print(f"  ⚠️ Checkpoint not found at {args.model}")
        print(f"     Running with untrained weights for demonstration.")

    model.eval()

    # ── Tokenizers ──
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")
    nlp = spacy.load("en_core_web_sm")

    # ── Load Data ──
    print(f"\n📥 Loading data from: {args.data}")
    if os.path.exists(args.data):
        df = pd.read_parquet(args.data)
        df = df[df["articles_clean"].str.len() > 100]
        df = df[df["summary_clean"].str.len() > 20]
        test_start = int(len(df) * 0.9)
        test_df = df.iloc[test_start:test_start + args.samples]
    else:
        print(f"  ⚠️ Data not found. Using dummy samples.")
        test_df = pd.DataFrame({
            "articles_clean": ["Bajaj Hindusthan approved a merger with Bajaj Eco-Tec."] * 5,
            "summary_clean": ["Bajaj Hindusthan approved a merger."] * 5,
        })

    print(f"  Evaluating on {len(test_df)} samples\n")

    # ── Generate Summaries ──
    print(f"{'='*60}")
    print("🚀 Generating Summaries")
    print(f"{'='*60}")

    predictions, references, sources = [], [], []
    latencies = []
    generated_tokens = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
        try:
            article = row["articles_clean"]
            reference = row["summary_clean"]

            # Sentence splitting for GAT graph
            sents = split_into_sentences(article)[:args.max_sents]
            if not sents:
                continue

            padded = sents + [""] * (args.max_sents - len(sents))
            joined_raw = "|||".join(padded)

            enc = bert_tok(padded, padding="max_length", truncation=True,
                           max_length=args.max_sent_len, return_tensors="pt")
            batch = {
                "sent_input_ids": enc["input_ids"].unsqueeze(0).to(device),
                "sent_attention_mask": enc["attention_mask"].unsqueeze(0).to(device),
                "sentences_raw": [joined_raw],
            }

            t0 = time.perf_counter()
            with torch.no_grad():
                gen_ids = model.generate_summary(
                    batch, num_beams=args.num_beams, max_length=args.max_summary_len
                )
            t1 = time.perf_counter()

            pred = bart_tok.decode(gen_ids[0], skip_special_tokens=True).strip()

            latencies.append(t1 - t0)
            generated_tokens += len(gen_ids[0])
            predictions.append(pred)
            references.append(reference)
            sources.append(article)

        except Exception as e:
            print(f"  ⚠️ Error: {e}")

    if not predictions:
        print("❌ No predictions generated. Aborting.")
        return

    print(f"✅ Generated {len(predictions)} summaries\n")

    # ── ROUGE & BERTScore ──
    print(f"{'='*60}")
    print("📊 Computing Metrics")
    print(f"{'='*60}")

    rouge = hf_evaluate.load("rouge")
    r = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    bleu = hf_evaluate.load("sacrebleu")
    bleu_out = bleu.compute(predictions=predictions,
                            references=[[ref] for ref in references])

    bertscore = hf_evaluate.load("bertscore")
    b = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_f1 = float(np.mean(b["f1"]))

    print(f"\n  ROUGE-1:      {r['rouge1']:.4f}")
    print(f"  ROUGE-2:      {r['rouge2']:.4f}")
    print(f"  ROUGE-L:      {r['rougeL']:.4f}")
    print(f"  SacreBLEU:    {bleu_out['score']:.2f}")
    print(f"  BERTScore F1: {bertscore_f1:.4f}")

    # ── Factual Consistency ──
    print(f"\n{'='*60}")
    print("🔍 Entity-Based Factual Consistency")
    print(f"{'='*60}")

    entity_results = []
    for src, gen in zip(sources, predictions):
        src_ents = extract_entities(src)
        gen_ents = extract_entities(gen)
        hallucinated = gen_ents - src_ents

        if gen_ents:
            precision = len(gen_ents & src_ents) / len(gen_ents)
        else:
            precision = 1.0

        entity_results.append({
            "precision": precision,
            "n_gen_entities": len(gen_ents),
            "n_hallucinated": len(hallucinated),
        })

    avg_precision = np.mean([r["precision"] for r in entity_results])
    total_halluc = sum(r["n_hallucinated"] for r in entity_results)
    total_ents = sum(r["n_gen_entities"] for r in entity_results)
    halluc_rate = total_halluc / max(total_ents, 1)

    print(f"  Entity Precision:     {avg_precision:.4f}")
    print(f"  Hallucination Rate:   {halluc_rate:.4f} ({total_halluc}/{total_ents})")

    # ── Inference Costs ──
    print(f"\n{'='*60}")
    print("⚡ Inference Costs")
    print(f"{'='*60}")

    avg_latency = np.mean(latencies)
    throughput = generated_tokens / sum(latencies) if latencies else 0

    print(f"  Latency:     {avg_latency:.3f} sec/summary")
    print(f"  Throughput:  {throughput:.1f} tokens/sec")

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  Peak VRAM:   {peak_vram:.2f} GB")

    # ── Qualitative Examples ──
    print(f"\n{'='*60}")
    print("👁️ Sample Outputs (first 3)")
    print(f"{'='*60}")

    for i in range(min(3, len(predictions))):
        print(f"\n{'─'*60}")
        print(f"  REF: {references[i][:250]}")
        print(f"  GEN: {predictions[i][:250]}")

    # ── Save Report ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = {
        "model": "HiGS (Dual-Encoder Fusion)",
        "checkpoint": os.path.basename(args.model),
        "num_samples": len(predictions),
        "scores": {
            "rouge1": round(r["rouge1"], 4),
            "rouge2": round(r["rouge2"], 4),
            "rougeL": round(r["rougeL"], 4),
            "sacrebleu": round(bleu_out["score"], 2),
            "bertscore_f1": round(bertscore_f1, 4),
        },
        "factual_consistency": {
            "entity_precision": round(avg_precision, 4),
            "hallucination_rate": round(halluc_rate, 4),
        },
        "inference": {
            "mean_latency_sec": round(float(avg_latency), 4),
            "throughput_tokens_per_sec": round(float(throughput), 1),
            "peak_vram_gb": round(peak_vram, 2) if peak_vram else None,
        },
        "system": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "device": str(device),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Report saved to: {args.output}")
    print(f"\n{'='*60}")
    print("✅ Evaluation Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
