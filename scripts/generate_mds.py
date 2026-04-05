"""
Multi-Document Summary Generator
=================================

Generates an abstractive summary from multiple articles using the HiGS model.
Works out-of-the-box — no path editing required.

Usage:
    python scripts/generate_mds.py
    python scripts/generate_mds.py --model model/higs_model.pt
"""

import os
import sys
import torch
import spacy
import argparse
from pathlib import Path
from transformers import AutoTokenizer, BartTokenizer

# ── AUTO-DISCOVER PROJECT ROOT ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def find_model_checkpoint():
    """Search common locations for the HiGS model checkpoint."""
    candidates = [
        PROJECT_ROOT / "model" / "higs_model.pt",
        PROJECT_ROOT / "data" / "HiGS" / "higs_model.pt",
        PROJECT_ROOT / "higs_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


# ── SAMPLE ARTICLES ─────────────────────────────────────────────────
# Edit these to test with your own articles on the same topic.

ARTICLE_1 = """
Tata Motors on Friday reported a 4% increase in global sales, including that of Jaguar Land Rover, at 1,02,396 units in November. Global wholesale units of all Tata Motors' commercial vehicles and Tata Daewoo range in November stood at 33,835 units, a growth of 1% over November 2013, the company said in a statement.
"""

ARTICLE_2 = """
Shares of Tata Motors surged today as the automobile giant announced solid global wholesale numbers. The growth was led by strong performance in its commercial vehicle segment, which saw a modest 1% uptick. Jaguar Land Rover (JLR) sales specifically showed signs of stabilization, contributing heavily to the total 1,02,396 units sold globally in the month of November.
"""

ARTICLE_3 = """
In an optimistic statement to investors, Tata Motors highlighted its November global sales figures crossing the 1 lakh unit mark. The 4 percent year-on-year growth represents a strong recovery phase for the company, particularly supported by consistent demand in its heavy commercial vehicle sectors and JLR divisions despite global market headwinds.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate multi-document summary")
    parser.add_argument("--model", default=None,
                        help="Path to model checkpoint (auto-detected if not specified)")
    parser.add_argument("--max-sents", type=int, default=30,
                        help="Maximum sentences to process (default: 30)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum summary length in tokens (default: 128)")
    parser.add_argument("--num-beams", type=int, default=4,
                        help="Beam search width (default: 4)")
    args = parser.parse_args()

    # Auto-discover model path
    model_path = args.model if args.model else find_model_checkpoint()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. LOAD MODELS ───────────────────────────────────────────
    print("🧠 Loading spaCy sentence chunker...")
    nlp = spacy.load("en_core_web_sm")

    print("📦 Loading Tokenizers...")
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")

    print(f"🚀 Loading HiGS Model from: {model_path}")
    model = HiGraphSum().to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = remap_checkpoint_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        print("  ✅ Model loaded successfully!")
    else:
        print(f"  ⚠️ Checkpoint not found at: {model_path}")
        print(f"     Download it from Google Drive and place it at: model/higs_model.pt")
        return

    model.eval()
    print("✅ Systems Ready!\n")

    # ── 2. PROCESS ARTICLES ──────────────────────────────────────
    articles = [ARTICLE_1, ARTICLE_2, ARTICLE_3]

    print(f"{'='*60}")
    print("🔍 PROCESSING DOCUMENTS FOR GRAPH NETWORK")
    print(f"{'='*60}")

    combined_raw_text = " ".join(articles)
    valid_sentences = split_into_sentences(combined_raw_text)[:args.max_sents]

    if not valid_sentences:
        print("❌ Error: No valid sentences found in articles.")
        return

    print(f"📊 Found {len(valid_sentences)} valid sentences across the articles.")
    print("   Building Adjacency Graph...")

    padded_sentences = valid_sentences + [""] * (args.max_sents - len(valid_sentences))
    joined_for_bart = "|||".join(padded_sentences)

    enc = bert_tok(
        padded_sentences,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    batch = {
        "sent_input_ids": enc["input_ids"].unsqueeze(0).to(device),
        "sent_attention_mask": enc["attention_mask"].unsqueeze(0).to(device),
        "sentences_raw": [joined_for_bart]
    }

    # ── 3. GENERATION ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("✨ GENERATING MULTI-DOCUMENT SUMMARY ✨")
    print(f"{'='*60}")

    with torch.no_grad():
        gen_ids = model.generate_summary(
            batch, num_beams=args.num_beams, max_length=args.max_length
        )
        summary = bart_tok.decode(gen_ids[0], skip_special_tokens=True).strip()

    print(f"\n{summary}\n")

    # ── 4. FACT CHECKING ─────────────────────────────────────────
    print(f"{'='*60}")
    print("🕵️ FACT CHECKING")
    print(f"{'='*60}")

    src_ents = extract_entities(combined_raw_text)
    gen_ents = extract_entities(summary)
    hallucinated = gen_ents - src_ents

    if not gen_ents:
        precision = 100.0
    else:
        precision = len(gen_ents & src_ents) / len(gen_ents) * 100

    print(f"Entities in Original Articles: {src_ents}")
    print(f"Entities in Final Summary:     {gen_ents}")
    print(f"🚨 Hallucinations:             {hallucinated if hallucinated else 'None ✅'}")
    print(f"🎯 Precision Score:            {precision:.1f}%")


if __name__ == "__main__":
    main()
