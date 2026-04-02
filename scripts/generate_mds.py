import os
import torch
import spacy
from transformers import AutoTokenizer, BartTokenizer
import sys

# ── CONFIGURATION & PATHS ──────────────────────────────────────────
BASE_DIR = r"C:\Shuvidha Foundation\HiGS_Multi_document_abstract_summarization_in_Indian_English"
CKPT_PATH = os.path.join(BASE_DIR, "model", "higs_model.pt")
sys.path.append(os.path.join(BASE_DIR, "scripts"))

# Import architectural definitions from existing runner
from run_local_consistency_check import HiGraphSum, remap_checkpoint_keys, extract_entities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. LOAD MODELS ───────────────────────────────────────────────
print("🧠 Loading spaCy sentence chunker...")
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text, min_len=10):
    """Splits a massive text block into clean, individual sentences for the Graph Model."""
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > min_len]

print("📦 Loading Tokenizers...")
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")

print(f"🚀 Loading HiGS Model from {os.path.basename(CKPT_PATH)}...")
model = HiGraphSum().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(remap_checkpoint_keys(ckpt["model_state_dict"]), strict=False)
model.eval()
print("✅ Systems Ready!\n")

# ── 2. YOUR MULTI-DOCUMENT INPUT ─────────────────────────────────
# Paste as many articles as you want in this list. They will be merged and summarized.
ARTICLE_1 = """
Tata Motors on Friday reported a 4% increase in global sales, including that of Jaguar Land Rover, at 1,02,396 units in November. Global wholesale units of all Tata Motors’ commercial vehicles and Tata Daewoo range in November stood at 33,835 units, a growth of 1% over November 2013, the company said in a statement.
"""

ARTICLE_2 = """
Shares of Tata Motors surged today as the automobile giant announced solid global wholesale numbers. The growth was led by strong performance in its commercial vehicle segment, which saw a modest 1% uptick. Jaguar Land Rover (JLR) sales specifically showed signs of stabilization, contributing heavily to the total 1,02,396 units sold globally in the month of November.
"""

ARTICLE_3 = """
In an optimistic statement to investors, Tata Motors highlighted its November global sales figures crossing the 1 lakh unit mark. The 4 percent year-on-year growth represents a strong recovery phase for the company, particularly supported by consistent demand in its heavy commercial vehicle sectors and JLR divisions despite global market headwinds.
"""

articles = [ARTICLE_1, ARTICLE_2, ARTICLE_3]

# ── 3. PROPER GRAPH PROCESSING ───────────────────────────────────
print(f"{'='*60}")
print("🔍 PROCESSING DOCUMENTS FOR GRAPH NETWORK")
print(f"{'='*60}")

# Step A: Combine entirely into one massive string
combined_raw_text = " ".join(articles)

# Step B: Split into literal sentences so the GAT matrix works!
valid_sentences = split_into_sentences(combined_raw_text)[:30] # Max 30 nodes for memory

if not valid_sentences:
    print("❌ Error: No valid sentences found in articles.")
    sys.exit()

print(f"📊 Found {len(valid_sentences)} valid sentences across the articles.")
print("   Building Adjacency Graph...")

# Step C: Pad to 30 nodes so model shapes match
padded_sentences = valid_sentences + [""] * (30 - len(valid_sentences))
joined_for_bart = "|||".join(padded_sentences)

# Step D: Tokenize each sentence completely individually (Max 64 words *per sentence*)
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


# ── 4. GENERATION ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("✨ GENERATING MULTI-DOCUMENT SUMMARY ✨")
print(f"{'='*60}")

with torch.no_grad():
    gen_ids = model.generate_summary(batch, num_beams=4, max_length=128)
    summary = bart_tok.decode(gen_ids[0], skip_special_tokens=True).strip()

print(f"\n{summary}\n")

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
