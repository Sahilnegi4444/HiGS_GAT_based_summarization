# HiGS: Hierarchical Graph-Based Summarization for Multi-Document Abstractive Summarization in Indian English

**A Dual-Encoder Fusion Architecture combining Graph Attention Networks with BART for factually consistent, parameter-efficient summarization of Indian English news.**

> **What is Multi-Document Abstractive Summarization?** Given multiple news articles covering the same event, the model generates a single, concise summary that captures the key facts from all articles — using its own words rather than simply copying sentences (abstractive vs. extractive).

---

## Key Results

**Metrics Explained:**
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — Measures word-level overlap between the generated summary and the human reference. ROUGE-1 counts matching single words (unigrams), ROUGE-2 counts matching word pairs (bigrams), and ROUGE-L finds the longest common subsequence. Higher is better.
- **BERTScore** — Measures semantic similarity between generated and reference text using deep contextual embeddings (DeBERTa model). Unlike ROUGE, it captures meaning even when different words are used. Range: 0–1, higher is better.

**Training Types:**
- **From-scratch** — The model was trained entirely on the NewsSumm dataset from random or pretrained initialization, with all parameters updated.
- **PEFT (Parameter-Efficient Fine-Tuning)** — Only a small subset of the model's parameters are trained (using LoRA adapters), keeping the rest frozen. This makes training feasible for very large models (7B+ parameters) on limited GPU memory.
- **Few-shot** — No training at all. The model is given a few example article→summary pairs in the prompt and asked to generate a summary for a new article.

| Model | Params | Training | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|-------|--------|----------|---------|---------|---------|-----------|
| PRIMERA | ~150M | From-scratch | 0.3231 | 0.1822 | 0.2634 | 0.8738 |
| LED-base | ~162M | From-scratch | 0.3222 | 0.1983 | 0.2878 | 0.8733 |
| LongT5-base | ~248M | From-scratch | 0.2658 | 0.1162 | 0.2055 | 0.7806 |
| Flan-T5-XL | ~3B | From-scratch | 0.2768 | 0.1475 | 0.2510 | 0.8476 |
| Flan-T5-XXL | ~11B | PEFT | 0.2760 | 0.1620 | 0.2170 | 0.7990 |
| Mistral-7B | ~7B | PEFT | 0.4072 | 0.2072 | 0.3013 | 0.8974 |
| Llama-3-8B | ~8B | PEFT | 0.2854 | 0.1862 | 0.3892 | 0.8680 |
| Qwen2-7B | ~7B | PEFT | 0.2734 | 0.1341 | 0.3840 | 0.8720 |
| Gemma-2-9B | ~9B | PEFT | 0.3418 | 0.1290 | 0.2308 | 0.8832 |
| Mixtral-8x7B | ~47B | Few-shot | 0.1787 | 0.1622 | 0.3770 | 0.8792 |
| **HiGS (Ours)** | **~250M** | **From-scratch** | **0.4888** | **0.3181** | **0.3739** | **0.8937** |

> **HiGS achieves the highest ROUGE-1 (0.4888) and ROUGE-2 (0.3181) across all models** — including 7B+ LLMs — while using only ~250M parameters (28–188× fewer than the LLM baselines).

### Factual Consistency (50-sample test set)

These metrics measure whether the generated summaries are **factually faithful** to the source articles:

| Metric | Score | What It Measures |
|--------|-------|------------------|
| Entity Precision | 0.9062 | Of all named entities (people, organizations, places) in the generated summary, 90.6% were actually present in the source articles. Higher is better. |
| Hallucination Rate | 0.10 | Only 10% of generated entities were "hallucinated" (fabricated by the model and not in the source). Lower is better. |
| NLI Entailment | 0.7833 | Using a DeBERTa-based Natural Language Inference (NLI) model, 78.3% of generated sentences are logically entailed (supported) by the source text. Higher is better. |
| NLI Contradiction | 0.0187 | Only 1.87% of generated sentences contradict the source, indicating very low factual conflict. Lower is better. |
| Topic Overlap | 0.8689 | 86.9% keyword overlap between source and summary topics, confirming the model stays on-topic and does not drift. Higher is better. |

---

## Quick Start (For Supervisors)

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Evaluate the model (scores, factual consistency, inference costs)
python scripts/evaluate_higs.py --model data/HiGS/higs_model.pt

# 3. Generate a multi-document summary
python scripts/generate_mds.py

# 4. Run full factual consistency check (entity + NLI + topic)
python scripts/run_local_consistency_check.py
```

---

## Architecture — Dual-Encoder Fusion

**Key Components:**
- **BERT (Bidirectional Encoder Representations from Transformers)** — A pretrained language model that reads text and converts each sentence into a dense numerical vector (embedding) capturing its meaning.
- **GAT (Graph Attention Network)** — A neural network that operates on graph structures. It takes the sentence embeddings, connects them via an **Adjacency Matrix** (a grid showing which sentences are related based on shared entities or semantic similarity), and uses attention to determine which connections are most important.
- **BART (Bidirectional and Auto-Regressive Transformer)** — A sequence-to-sequence model with its own encoder and decoder. Its decoder generates fluent English text word-by-word.
- **MLP Projection** — A small feed-forward network (`Linear → GELU → LayerNorm`) that transforms the GAT's 512-dimensional output to match BART's expected 768-dimensional input space.
- **Tokenizer** — Converts raw text to numerical token IDs that the model can process. BERT and BART use different vocabularies, so HiGS uses **dual tokenizers**.

HiGS uses two parallel encoder pathways fused before decoding:

```
                     ┌─────────────────────────────────────────┐
                     │  Pathway B: Native BART Encoder (frozen) │
                     │  Input → BART Tokenizer → BART Encoder   │
                     │  Output: z_sem (R^768)                   │
                     └──────────────────┬──────────────────────┘
                                        │
Input Document  ────────────────────────┤──── Concat ──── BART Decoder ──── Summary
Cluster                                 │
                     ┌──────────────────┴──────────────────────┐
                     │  Pathway A: Structural Graph Modeling     │
                     │  Input → BERT [CLS] → Adjacency Matrix   │
                     │  → GAT Layers → MLP Projection + LayerNorm│
                     │  Output: z_struct (R^768)                 │
                     └─────────────────────────────────────────┘
```

**How it works:**
1. **Pathway A (Structure):** Sentences are fed through BERT, which extracts a `[CLS]` embedding (a single vector summarizing each sentence). These embeddings are connected via an Adjacency Matrix (built from cosine similarity + named entity overlap) and processed by 2 GAT layers to produce graph-enriched representations. An MLP Projection block maps these from 512-dim to 768-dim.
2. **Pathway B (Semantics):** The same raw text is simultaneously fed through BART's own native encoder, producing word-level embeddings the decoder natively understands.
3. **Fusion:** Both outputs are concatenated and passed to the BART Decoder, which generates the summary word-by-word using **beam search** (exploring multiple candidate sequences in parallel and selecting the highest-probability path).

**Why Dual-Encoder?** Standard models use a single encoder — if only BART, they lose cross-document factual alignment; if only BERT+GAT, they produce broken output because the BART decoder receives embeddings from an incompatible **latent space** (the mathematical coordinate system the model works in). Our Dual-Encoder Fusion ensures the decoder always has native BART representations for fluency, augmented with graph-enriched structural representations for factual grounding.

See [docs/architecture.md](docs/architecture.md) for the full mathematical formulation.

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── higs.yaml                   # HiGS training configuration
│   ├── primera.yaml
│   ├── led_baseline.yaml
│   ├── longt5.yaml
│   ├── flan_t5_xl.yaml
│   └── llm_lora.yaml
├── data/
│   ├── README.md
│   └── newssumm_cleaned.parquet    # (download separately)
├── model/
│   └── higs_model.pt              # Trained checkpoint (download separately)
├── models/
│   ├── higs_model.py              # HiGS architecture (Dual-Encoder Fusion)
│   ├── baselines.py               # Encoder-decoder baseline loader
│   └── llm_lora.py                # LLM LoRA/QLoRA wrapper
├── scripts/
│   ├── prepare_data.py            # Data cleaning, splitting, and EDA stats
│   ├── train_higs.py              # HiGS two-phase training pipeline
│   ├── train_baselines.py         # Unified baseline training (enc-dec & LLM LoRA)
│   ├── evaluate_higs.py           # HiGS evaluation (scores + consistency metrics)
│   ├── run_local_consistency_check.py  # Full factual consistency pipeline
│   └── generate_mds.py            # Multi-document summary generation
├── notebooks/
│   └── dataset_analysis.ipynb     # Dataset EDA & visualization
├── results/
│   ├── benchmark_table.csv              # Final benchmark scores (TABLE III format)
│   ├── comprehensive_model_comparison.csv # Synthesized evaluation of all models
│   ├── eval_reports/                    # Detailed individual model evaluation JSONs
│   ├── benchmark_reports/               # Detailed inference benchmark JSONs
│   ├── factual_consistency_report.json  # HiGS consistency metrics
│   └── generated_samples.json           # Qualitative generated outputs
└── docs/
    ├── architecture.md            # Full architecture specification
    └── higs_architecture.png      # Architecture diagram
```

> **Only 6 scripts** — each with a clear, non-overlapping purpose. No redundant files.

---

## Setup

### Requirements

- Python 3.10+
- CUDA 11.8+ (NVIDIA's GPU computing toolkit, required for GPU-accelerated training)
- PyTorch 2.0+ (deep learning framework)
- 16 GB VRAM (GPU video memory, for training) / 8 GB VRAM (for inference only)
- **spaCy** (`en_core_web_sm`) — NLP library used for sentence splitting and named entity extraction

### Installation

```bash
git clone https://github.com/Sahilnegi4444/HiGS_GAT_based_summarization.git
cd HiGS_Multi_document_abstract_summarization_in_Indian_English

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## Data Pipeline

All data preparation is handled by a single script with step selection:

### Run Full Pipeline (clean → split → stats)

```bash
python scripts/prepare_data.py
```

### Or Run Individual Steps

```bash
# Step 1: Clean raw Excel → Parquet (HTML stripping, deduplication, token counting)
python scripts/prepare_data.py --step clean

# Step 2: Create 80/10/10 train/val/test splits
python scripts/prepare_data.py --step split

# Step 3: Compute statistics & generate EDA plots
python scripts/prepare_data.py --step stats
```

---

## Training HiGS

HiGS uses a **two-phase training strategy** to prevent the graph components and decoder from interfering with each other during learning:

### Phase 1 — Train the Graph (20 epochs, lr=3e-5)

Trains the BERT encoder, GAT layers, and MLP Projection to learn structural graph routing. The BART encoder and decoder are **frozen** (their weights are locked and not updated), so the decoder's pretrained language ability is preserved.

```bash
python scripts/train_higs.py --config configs/higs.yaml --phase 1
```

### Phase 2 — Align the Decoder (10 epochs, lr=1e-5)

Freezes BERT + GAT (already trained). Unfreezes the BART decoder so it can learn to generate summaries conditioned on the now-stable graph-enriched representations.

```bash
python scripts/train_higs.py --config configs/higs.yaml --phase 2 \
    --checkpoint results/higs/best_checkpoint.pt
```

### Resume Training

If training is interrupted, resume from any saved checkpoint:

```bash
python scripts/train_higs.py --config configs/higs.yaml --phase 2 \
    --checkpoint results/higs/checkpoint_step10000.pt --resume
```

---

## Evaluation

### HiGS Model (Scores + Factual Consistency + Inference Costs)

```bash
python scripts/evaluate_higs.py --model data/HiGS/higs_model.pt --samples 50
```

Reports:
- **Text Quality:** ROUGE-1/2/L (n-gram overlap), SacreBLEU (translation-style precision metric), BERTScore (semantic similarity)
- **Factual Consistency:** Entity Precision (are the names/places in the summary real?), Hallucination Rate (% of fabricated entities)
- **Inference Costs:** Latency (seconds per summary), Throughput (tokens per second), Peak VRAM (GPU memory used)

### Full Factual Consistency Check

```bash
python scripts/run_local_consistency_check.py
```

Runs a three-part analysis:
1. **Entity-Based** — Checks if named entities in the summary exist in the source articles (using spaCy NER)
2. **NLI-Based** — Uses a DeBERTa Natural Language Inference model to check if each generated sentence is logically entailed, neutral, or contradicted by the source
3. **Topic Consistency** — Measures keyword overlap to ensure the summary stays on-topic

---

## Multi-Document Inference

Generate a summary from multiple articles on the same topic:

```bash
python scripts/generate_mds.py
```

Edit the `ARTICLE_1`, `ARTICLE_2`, `ARTICLE_3` variables in the script to test with your own articles.

---

## Baseline Models

All baselines (encoder-decoder and LLM LoRA) are trained with a **single unified script** that auto-detects the model type:

### Encoder-Decoder (From-Scratch)

These smaller models (150M–3B params) are fully trained on the NewsSumm dataset:

```bash
python scripts/train_baselines.py --config configs/primera.yaml
python scripts/train_baselines.py --config configs/led_baseline.yaml
python scripts/train_baselines.py --config configs/longt5.yaml
python scripts/train_baselines.py --config configs/flan_t5_xl.yaml
```

### LLMs with LoRA/PEFT

**LoRA (Low-Rank Adaptation)** — Instead of training all 7+ billion parameters (which would require massive GPU memory), LoRA injects small trainable adapter matrices into the model while keeping the original weights frozen. This reduces trainable parameters from billions to just a few million, making fine-tuning feasible on consumer GPUs.

```bash
python scripts/train_baselines.py --config configs/llm_lora.yaml --model mistralai/Mistral-7B-Instruct-v0.3
python scripts/train_baselines.py --config configs/llm_lora.yaml --model meta-llama/Meta-Llama-3-8B-Instruct
python scripts/train_baselines.py --config configs/llm_lora.yaml --model Qwen/Qwen2-7B-Instruct
python scripts/train_baselines.py --config configs/llm_lora.yaml --model google/gemma-2-9b-it
```

---

## Acknowledgment

This research was conducted under the Internship Program at **Suvidha Foundation (Suvidha Mahila Mandal)**, Nagpur, India. The author gratefully acknowledges the Research Mentorship Support provided throughout the development of this project.

---

## Citation

```bibtex
@article{negi2026higs,
  title={HiGS: A Multi-Document Abstract Summarization Using Graph Attention Network},
  author={Negi, Sahil},
  year={2026},
  institution={Suvidha Foundation (Suvidha Mahila Mandal), Nagpur, India}
}
```

---

## License

This project is released for academic research purposes.
