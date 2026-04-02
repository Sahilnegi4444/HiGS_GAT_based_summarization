"""
Local Factual Consistency Checker for HiGS Summaries
=====================================================
Adapted for local GPU execution (no Colab/Kaggle).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    BartForConditionalGeneration, BartTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import spacy
import json
import os
import sys

# ── CONFIG ──────────────────────────────────────────────────────────
BASE_DIR = r"C:\Shuvidha Foundation\HiGS_Multi_document_abstract_summarization_in_Indian_English"
DATA_PATH = os.path.join(BASE_DIR, "data", "newssumm_cleaned.parquet")
CKPT_PATH = os.path.join(BASE_DIR, "model", "higs_model.pt")
SAVE_PATH = os.path.join(BASE_DIR, "results", "factual_consistency_report.json")
SAMPLES_PATH = os.path.join(BASE_DIR, "results", "generated_samples.json")

NUM_SAMPLES = 50
MAX_SENTS = 30
MAX_SENT_LEN = 64
MAX_SUMMARY_LEN = 128
NUM_BEAMS = 4

# ── SETUP ───────────────────────────────────────────────────────────
print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── NLP UTILITIES ───────────────────────────────────────────────────

def split_into_sentences(text, min_len=10):
    doc = nlp(text[:10000])
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > min_len]

def extract_entities(text):
    if not text:
        return set()
    doc = nlp(text[:5000])
    return {ent.text.lower() for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "MONEY"}}

def build_adjacency_matrix(sentences, sent_embeddings, threshold=0.75):
    valid_idx = [i for i, s in enumerate(sentences) if s.strip()]
    n_total = sent_embeddings.size(0)
    adj = torch.zeros(n_total, n_total)
    if len(valid_idx) > 1:
        valid_emb = sent_embeddings[valid_idx]
        normed = F.normalize(valid_emb, p=2, dim=1)
        sim = torch.mm(normed, normed.t())
        for ii, i in enumerate(valid_idx):
            for jj, j in enumerate(valid_idx):
                if sim[ii, jj] > threshold:
                    adj[i, j] = 1.0
        entities = [extract_entities(sentences[i]) for i in valid_idx]
        for ii, i in enumerate(valid_idx):
            for jj, j in enumerate(valid_idx):
                if ii < jj and (entities[ii] & entities[jj]):
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)
    return adj

# ── MODEL ───────────────────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)
        B, N, _ = Wh.size()
        Wi = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wj = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leakyrelu(self.a(torch.cat([Wi, Wj], dim=-1)).squeeze(-1))
        e = e.masked_fill(adj == 0, float("-inf"))
        att = torch.nan_to_num(F.softmax(e, dim=-1), nan=0.0)
        return torch.bmm(self.dropout_layer(att), Wh)

class HiGraphSum(nn.Module):
    def __init__(self, num_gat_layers=2, gat_hidden_dim=512, dropout=0.2, label_smoothing=0.1):
        super().__init__()
        self.sentence_encoder = AutoModel.from_pretrained("bert-base-uncased")
        bert_dim = self.sentence_encoder.config.hidden_size
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(bert_dim if i == 0 else gat_hidden_dim, gat_hidden_dim, dropout)
            for i in range(num_gat_layers)
        ])
        self.gat_dropout = nn.Dropout(dropout)
        self.decoder = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        bart_hidden_dim = self.decoder.config.d_model
        
        self.projection = nn.Sequential(
            nn.Linear(gat_hidden_dim, bart_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(bart_hidden_dim)
        )
        
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def encode_sentences(self, ids, mask):
        B, S, L = ids.size()
        out = self.sentence_encoder(input_ids=ids.view(-1, L), attention_mask=mask.view(-1, L))
        cls = out.last_hidden_state[:, 0, :]
        sent_embeddings = cls.view(B, S, -1)
        word_embeddings = out.last_hidden_state.view(B, S * L, -1)
        return sent_embeddings, word_embeddings

    @torch.no_grad()
    def generate_summary(self, batch, num_beams=4, max_length=128):
        ids = batch["sent_input_ids"]
        mask = batch["sent_attention_mask"]
        raw = batch["sentences_raw"]
        B = ids.size(0)

        node_features, word_features = self.encode_sentences(ids, mask)
        adjs = []
        for i in range(B):
            s = raw[i].split("|||") if isinstance(raw[i], str) else list(raw[i])
            adjs.append(build_adjacency_matrix(s, node_features[i], 0.75))
        adj = torch.stack(adjs).to(node_features.device)

        h = node_features
        for gat in self.gat_layers:
            h = F.relu(gat(h, adj))
            h = self.gat_dropout(h)

        sentence_hidden = self.projection(h)
        
        source_texts = [s.replace("|||", " ") for s in raw]
        bart_inputs = self.bart_tokenizer(
            source_texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(sentence_hidden.device)
        
        bart_encoder_outputs = self.decoder.model.encoder(
            input_ids=bart_inputs.input_ids,
            attention_mask=bart_inputs.attention_mask,
            return_dict=True
        ).last_hidden_state
        
        encoder_hidden = torch.cat([bart_encoder_outputs, sentence_hidden], dim=1)
        enc_out = BaseModelOutput(last_hidden_state=encoder_hidden)

        sent_enc_mask = torch.zeros(B, h.size(1), dtype=torch.long, device=h.device)
        for i in range(B):
            sents_i = raw[i].split("|||") if isinstance(raw[i], str) else list(raw[i])
            n_real = len([s for s in sents_i if s.strip()])
            sent_enc_mask[i, :n_real] = 1
            
        enc_mask = torch.cat([bart_inputs.attention_mask, sent_enc_mask], dim=1)

        return self.decoder.generate(
            encoder_outputs=enc_out,
            attention_mask=enc_mask,
            num_beams=num_beams, max_length=max_length, early_stopping=True,
            no_repeat_ngram_size=3, length_penalty=2.0,
        )


# ── CHECKPOINT KEY REMAPPING ────────────────────────────────────────
def remap_checkpoint_keys(state_dict):
    """Remap old checkpoint keys to new architecture keys."""
    new_sd = {}
    for key, value in state_dict.items():
        if key == "projection.weight":
            new_sd["projection.0.weight"] = value
        elif key == "projection.bias":
            new_sd["projection.0.bias"] = value
        elif key in ("residual_gate",) or key.startswith("word_projection.0.") or key.startswith("word_projection.2."):
            continue  # skip removed/replaced params
        else:
            new_sd[key] = value
    return new_sd


# ── LOAD MODEL + DATA ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("📦 Loading model and data...")
print(f"{'='*60}")

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")

model = HiGraphSum().to(device)
print(f"Loading checkpoint: {os.path.basename(CKPT_PATH)}")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
state_dict = remap_checkpoint_keys(ckpt["model_state_dict"])
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"✅ Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
      f"train_loss={ckpt.get('train_loss', '?'):.4f}, val_loss={ckpt.get('val_loss', '?'):.4f}")

print(f"\nLoading data: {os.path.basename(DATA_PATH)}")
df = pd.read_parquet(DATA_PATH)
df = df[df["articles_clean"].str.len() > 100]
df = df[df["summary_clean"].str.len() > 20]
test_start = int(len(df) * 0.9)
test_df = df.iloc[test_start:test_start + NUM_SAMPLES]
print(f"✅ {len(test_df)} test samples selected (from index {test_start})")


# ── GENERATE SUMMARIES ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("🚀 Generating Summaries...")
print(f"{'='*60}")

sources, predictions, references = [], [], []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
    try:
        article = row["articles_clean"]
        sents = split_into_sentences(article)[:MAX_SENTS]
        if not sents:
            continue
        padded = sents + [""] * (MAX_SENTS - len(sents))
        enc = bert_tok(padded, padding="max_length", truncation=True,
                       max_length=MAX_SENT_LEN, return_tensors="pt")
        batch = {
            "sent_input_ids": enc["input_ids"].unsqueeze(0).to(device),
            "sent_attention_mask": enc["attention_mask"].unsqueeze(0).to(device),
            "sentences_raw": ["|||".join(padded)],
        }
        gen_ids = model.generate_summary(batch, num_beams=NUM_BEAMS, max_length=MAX_SUMMARY_LEN)
        pred = bart_tok.decode(gen_ids[0], skip_special_tokens=True)

        sources.append(article)
        predictions.append(pred.strip())
        references.append(row["summary_clean"])
    except Exception as e:
        print(f"\n⚠️ Error: {e}")

print(f"✅ Generated {len(predictions)} summaries")


# ── SHOW GENERATED SAMPLES ──────────────────────────────────────────
print(f"\n{'='*60}")
print("📝 GENERATED SAMPLES (first 10)")
print(f"{'='*60}")

samples_for_json = []
for i in range(min(10, len(predictions))):
    print(f"\n{'─'*60}")
    print(f"  SAMPLE {i+1}")
    print(f"  SRC: {sources[i][:300]}...")
    print(f"  REF: {references[i][:300]}")
    print(f"  GEN: {predictions[i][:300]}")
    samples_for_json.append({
        "id": i,
        "source": sources[i][:500],
        "reference": references[i],
        "generated": predictions[i],
    })


# ── ENTITY-BASED FACTUAL CONSISTENCY ────────────────────────────────
print(f"\n{'='*60}")
print("🔍 ENTITY-BASED FACTUAL CONSISTENCY")
print(f"{'='*60}")

entity_results = []
for src, gen, ref in zip(sources, predictions, references):
    src_ents = extract_entities(src)
    gen_ents = extract_entities(gen)
    ref_ents = extract_entities(ref)

    if gen_ents:
        ent_precision = len(gen_ents & src_ents) / len(gen_ents)
        hallucinated = gen_ents - src_ents
    else:
        ent_precision = 1.0
        hallucinated = set()

    if ref_ents:
        ent_recall = len(gen_ents & ref_ents) / len(ref_ents)
    else:
        ent_recall = 1.0

    ent_f1 = (2 * ent_precision * ent_recall / (ent_precision + ent_recall)
              if ent_precision + ent_recall > 0 else 0)

    entity_results.append({
        "precision": ent_precision,
        "recall": ent_recall,
        "f1": ent_f1,
        "n_gen_entities": len(gen_ents),
        "n_hallucinated": len(hallucinated),
        "hallucinated_entities": list(hallucinated)[:5],
    })

avg_prec = np.mean([r["precision"] for r in entity_results])
avg_rec = np.mean([r["recall"] for r in entity_results])
avg_f1 = np.mean([r["f1"] for r in entity_results])
total_halluc = sum(r["n_hallucinated"] for r in entity_results)
total_ents = sum(r["n_gen_entities"] for r in entity_results)
halluc_rate = total_halluc / max(total_ents, 1)

print(f"  Entity Precision (faithfulness): {avg_prec:.4f}")
print(f"  Entity Recall (coverage):        {avg_rec:.4f}")
print(f"  Entity F1:                       {avg_f1:.4f}")
print(f"  Hallucination Rate:              {halluc_rate:.4f} ({total_halluc}/{total_ents})")


# ── NLI-BASED FACTUAL CONSISTENCY ──────────────────────────────────
print(f"\n{'='*60}")
print("🧠 NLI-BASED FACTUAL CONSISTENCY (DeBERTa)")
print(f"{'='*60}")
print("  Loading NLI model...")

nli_model_name = "cross-encoder/nli-deberta-v3-base"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
nli_model.eval()

entailment_scores = []
contradiction_scores = []

for src, gen in tqdm(zip(sources, predictions), total=len(predictions), desc="NLI Check"):
    try:
        premise = src[:1000]
        hypothesis = gen
        enc = nli_tokenizer(premise, hypothesis, return_tensors="pt",
                           max_length=512, truncation=True).to(device)
        with torch.no_grad():
            logits = nli_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
        entailment_scores.append(probs[1].item())
        contradiction_scores.append(probs[0].item())
    except Exception as e:
        print(f"⚠️ {e}")

avg_entailment = np.mean(entailment_scores) if entailment_scores else 0
avg_contradiction = np.mean(contradiction_scores) if contradiction_scores else 0
n_entailed = sum(1 for s in entailment_scores if s > 0.5)
n_contradicted = sum(1 for s in contradiction_scores if s > 0.5)

print(f"\n  Avg Entailment Score:    {avg_entailment:.4f}")
print(f"  Avg Contradiction Score: {avg_contradiction:.4f}")
print(f"  Entailed summaries:     {n_entailed}/{len(entailment_scores)} "
      f"({100*n_entailed/max(len(entailment_scores),1):.0f}%)")
print(f"  Contradicted summaries: {n_contradicted}/{len(contradiction_scores)} "
      f"({100*n_contradicted/max(len(contradiction_scores),1):.0f}%)")

del nli_model, nli_tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── TOPIC CONSISTENCY ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("📋 TOPIC CONSISTENCY")
print(f"{'='*60}")

topic_scores = []
STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
              "to", "for", "of", "and", "or", "but", "has", "have", "had",
              "with", "from", "by", "that", "this", "it", "its", "be", "been"}

for src, gen in zip(sources, predictions):
    src_words = {w.lower() for w in src.split() if w.lower() not in STOP_WORDS and len(w) > 3}
    gen_words = {w.lower() for w in gen.split() if w.lower() not in STOP_WORDS and len(w) > 3}
    topic_overlap = len(gen_words & src_words) / len(gen_words) if gen_words else 0
    topic_scores.append(topic_overlap)

avg_topic = np.mean(topic_scores)
low_topic = sum(1 for s in topic_scores if s < 0.3)

print(f"  Avg Topic Overlap:      {avg_topic:.4f}")
print(f"  Off-topic summaries:    {low_topic}/{len(topic_scores)} (overlap < 0.3)")


# ── WORST SAMPLES ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("⚠️  WORST 5 SAMPLES (highest hallucination)")
print(f"{'='*60}")

combined = []
for i in range(len(predictions)):
    combined.append({
        "idx": i,
        "entity_precision": entity_results[i]["precision"],
        "entailment": entailment_scores[i] if i < len(entailment_scores) else 0,
        "topic_overlap": topic_scores[i],
        "hallucinated": entity_results[i]["hallucinated_entities"],
    })

combined.sort(key=lambda x: x["entity_precision"])

for item in combined[:5]:
    i = item["idx"]
    print(f"\n{'─'*60}")
    print(f"  Sample {i} | EntPrec: {item['entity_precision']:.2f} | "
          f"NLI: {item['entailment']:.2f} | Topic: {item['topic_overlap']:.2f}")
    print(f"  SRC: {sources[i][:200]}...")
    print(f"  GEN: {predictions[i][:200]}")
    print(f"  REF: {references[i][:200]}")
    if item["hallucinated"]:
        print(f"  🔴 Hallucinated: {item['hallucinated']}")


# ── FINAL REPORT ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("📊 FACTUAL CONSISTENCY REPORT")
print(f"{'='*60}")

report = {
    "model": "HiGS",
    "checkpoint": os.path.basename(CKPT_PATH),
    "num_samples": len(predictions),
    "entity_consistency": {
        "precision": round(avg_prec, 4),
        "recall": round(avg_rec, 4),
        "f1": round(avg_f1, 4),
        "hallucination_rate": round(halluc_rate, 4),
    },
    "nli_consistency": {
        "avg_entailment": round(avg_entailment, 4),
        "avg_contradiction": round(avg_contradiction, 4),
        "pct_entailed": round(100 * n_entailed / max(len(entailment_scores), 1), 1),
        "pct_contradicted": round(100 * n_contradicted / max(len(contradiction_scores), 1), 1),
    },
    "topic_consistency": {
        "avg_overlap": round(avg_topic, 4),
        "pct_off_topic": round(100 * low_topic / max(len(topic_scores), 1), 1),
    },
    "samples": samples_for_json,
}

if avg_prec > 0.7 and avg_entailment > 0.5 and avg_topic > 0.5:
    verdict = "✅ GOOD — Model is mostly factually consistent"
elif avg_prec > 0.5 and avg_entailment > 0.3:
    verdict = "⚠️ FAIR — Model has some hallucination issues"
else:
    verdict = "❌ POOR — Model is heavily hallucinating"

report["verdict"] = verdict

print(f"\n  {verdict}")
print(f"\n  Entity Precision:    {avg_prec:.4f}  {'✅' if avg_prec > 0.7 else '❌'}")
print(f"  Entailment Score:    {avg_entailment:.4f}  {'✅' if avg_entailment > 0.5 else '❌'}")
print(f"  Topic Overlap:       {avg_topic:.4f}  {'✅' if avg_topic > 0.5 else '❌'}")
print(f"  Hallucination Rate:  {halluc_rate:.4f}  {'✅' if halluc_rate < 0.2 else '❌'}")

# Save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n💾 Report saved to: {SAVE_PATH}")

# Save all generated samples
all_samples = []
for i in range(len(predictions)):
    all_samples.append({
        "id": i,
        "source_excerpt": sources[i][:500],
        "reference": references[i],
        "generated": predictions[i],
        "entity_precision": entity_results[i]["precision"],
        "hallucinated_entities": entity_results[i]["hallucinated_entities"],
        "entailment": entailment_scores[i] if i < len(entailment_scores) else None,
        "topic_overlap": topic_scores[i],
    })

with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)
print(f"💾 All samples saved to: {SAMPLES_PATH}")

print(f"\n{'='*60}")
print("✅ DONE")
print(f"{'='*60}")
