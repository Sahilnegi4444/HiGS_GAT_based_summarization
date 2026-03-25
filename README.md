# HiGS: Hierarchical Graph-Based Summarization for Multi-Document Abstractive Summarization in Indian English

**Reproducible Multi-Document Summarization Research Framework**
Indian English News | Long-Context Modeling | Hierarchical Graph Attention

---

## Overview

This repository provides a fully reproducible framework for benchmarking multi-document abstractive summarization systems on the **NewsSumm** dataset — a large-scale Indian English news corpus.

The project includes:
- End-to-end data cleaning and preprocessing pipelines
- Config-driven experiment tracking for reproducibility
- Training scripts for 10 baseline models (encoder-decoder + LLMs with LoRA)
- A novel **Hierarchical Graph-based Summarization (HiGS)** architecture
- Unified evaluation pipeline (ROUGE + BERTScore)
- Complete benchmark results

**HiGS** combines a BERT encoder with Graph Attention Network (GAT) layers and a BART decoder (~250M parameters), achieving a ROUGE-L of **0.2122** and BERTScore of **0.8466** with only ~250M parameters — demonstrating that explicit graph-based structural modeling provides a lightweight, efficient alternative to billion-parameter LLMs for multi-document summarization.

**Why HiGS?** Unlike 7B+ LLM baselines that require expensive cloud GPUs, HiGS runs inference on **standard 8GB RAM desktops/PCs** at ~45 tokens/sec — making it a practical, low-latency alternative for resource-constrained environments where deploying massive LLMs is infeasible.

> **Note:** Due to time and hardware constraints, HiGS training was not fully converged. The model exhibited steady convergence and scores are expected to improve significantly with extended training.

📄 **[Read the Full Research Paper on Overleaf](https://www.overleaf.com/read/ktqqqzvbcvmc#37caf1)**

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── higs.yaml
│   ├── primera.yaml
│   ├── led_baseline.yaml
│   ├── longt5.yaml
│   ├── flan_t5_xl.yaml
│   └── llm_lora.yaml
├── data/
│   └── README.md
├── models/
│   ├── higs_model.py            # HiGS architecture (BERT + GAT + BART)
│   ├── baselines.py             # Encoder-decoder baseline loader
│   └── llm_lora.py              # LLM LoRA/QLoRA wrapper
├── scripts/
│   ├── clean_dataset.py         # Data cleaning pipeline
│   ├── preprocess.py            # Preprocessing & train/val/test splits
│   ├── compute_stats.py         # Dataset statistics & EDA
│   ├── train_baseline.py        # Config-driven baseline training
│   ├── train_higs.py            # HiGS two-phase training
│   ├── train_llm_lora.py        # LLM LoRA fine-tuning
│   └── evaluate.py              # Unified evaluation (ROUGE + BERTScore)
├── notebooks/
│   ├── dataset_analysis.ipynb      # Preprocessing, feature engineering & data analysis
│   └── evaluate_higs_sample.ipynb  # Sample evaluation with outputs (~2 epochs)
├── results/
│   └── benchmark_table.csv      # Final benchmark scores
└── docs/
    ├── architecture.md              # HiGS architecture documentation
    └── higs_architecture.png        # Architecture pipeline diagram
```

---

## 1. System Requirements

### Hardware
**Minimum:**
- 1× GPU (16 GB VRAM recommended)
- 32 GB RAM
- 50 GB disk space

**For large models (Mixtral, Gemma-2, LLaMA-3):**
- 24–48 GB VRAM recommended

### Software
- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+

---

## 2. Environment Setup

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate
```bash
# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Add Hugging Face Token
```bash
export HF_TOKEN=your_hf_token_here
```

### Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 3. Dataset & Model Downloads

### Cleaned Dataset (Parquet)

Download the pre-cleaned dataset to skip the cleaning pipeline:

📥 **[Download Cleaned Dataset (Parquet)](https://drive.google.com/drive/folders/1l_5WC5gacZAnCjZCgcSC6ZvNc4Sa2Igc?usp=sharing)**

Place the downloaded `.parquet` file in:
```
data/processed/newssumm_cleaned.parquet
```

### Model Checkpoint

Download the trained HiGS model checkpoint to evaluate or fine-tune:

📥 **[Download HiGS Model Checkpoint](https://drive.google.com/drive/folders/1hqYPvjdl443WFcgfs9OA-73p0U5Nusbm?usp=sharing)**

Place the downloaded `.pt` file in:
```
results/higs/best_checkpoint.pt
```

### Raw Dataset (Optional)

If you want to run the full cleaning pipeline from scratch, place the raw Excel file in:
```
data/NewsSumm_Dataset.xlsx
```

> **Note:** The raw dataset is not included in this repository. Contact the dataset maintainers for access.

---

## 4. Data Cleaning Pipeline

```bash
python scripts/clean_dataset.py
```

This step:
- Removes missing article or summary rows
- Cleans HTML tags and markup
- Normalizes whitespace and removes duplicates
- Filters corrupted entries
- Standardizes column names

**Output:** `data/processed/newssumm_cleaned.parquet`

---

## 5. Preprocessing Pipeline

Convert cleaned data into processed parquet with train/val/test splits:

```bash
python scripts/preprocess.py \
    --input data/processed/newssumm_cleaned.parquet \
    --output data/processed
```

**Output:**
```
data/processed/
├── train.parquet
├── val.parquet
└── test.parquet
```

---

## 6. Dataset Statistics

Compute dataset diagnostics and generate visualizations:

```bash
python scripts/compute_stats.py \
    --data data/processed/newssumm_cleaned.parquet
```

Reports:
- Total articles, average tokens per article/summary
- Compression ratio statistics
- Category and source distributions
- Distribution plots saved to `data/processed/`

---

## 7. Baseline Models

### a. Encoder-Decoder Baselines

All encoder-decoder models are trained using config-driven scripts:

```bash
# LED (Longformer Encoder-Decoder)
python scripts/train_baseline.py --config configs/led_baseline.yaml

# LongT5
python scripts/train_baseline.py --config configs/longt5.yaml

# PRIMERA
python scripts/train_baseline.py --config configs/primera.yaml

# Flan-T5-XL
python scripts/train_baseline.py --config configs/flan_t5_xl.yaml
```

### b. LLM Baselines (LoRA/PEFT)

Fine-tune instruction-tuned LLMs with parameter-efficient methods:

```bash
# Mistral-7B-Instruct
python scripts/train_llm_lora.py --config configs/llm_lora.yaml --model mistralai/Mistral-7B-Instruct-v0.3

# LLaMA-3-8B-Instruct
python scripts/train_llm_lora.py --config configs/llm_lora.yaml --model meta-llama/Meta-Llama-3-8B-Instruct

# Qwen2-7B-Instruct
python scripts/train_llm_lora.py --config configs/llm_lora.yaml --model Qwen/Qwen2-7B-Instruct

# Gemma-2-9B-Instruct
python scripts/train_llm_lora.py --config configs/llm_lora.yaml --model google/gemma-2-9b-it

# Mixtral-8x7B-Instruct
python scripts/train_llm_lora.py --config configs/llm_lora.yaml --model mistralai/Mixtral-8x7B-Instruct-v0.1
```

---

## 8. Novel Model – HiGS (Hierarchical Graph-Based Summarization)

HiGS separates summarization into hierarchical stages:

1. **Sentence Encoding** — BERT encodes each sentence independently into dense representations
2. **Graph Construction** — Entity-overlap and cosine-similarity edges connect sentences across documents
3. **Graph Reasoning** — Multi-layer GAT aggregates cross-document information
4. **Conditional Decoding** — BART decoder generates the summary conditioned on graph-enriched representations

### Training

HiGS uses a two-phase training strategy:

```bash
# Phase 1: Full model training (encoder + GAT + decoder)
python scripts/train_higs.py --config configs/higs.yaml --phase 1

# Phase 2: Decoder-only fine-tuning with fixed encoder/GAT
python scripts/train_higs.py --config configs/higs.yaml --phase 2
```

### Architecture Details

See [docs/architecture.md](docs/architecture.md) for the full architecture specification, including:
- Mathematical formulation (encoder, GAT, projection, decoder)
- Architecture diagram
- Design justifications

---

## 9. Evaluation

Evaluate any model's predictions:

```bash
python scripts/evaluate.py \
    --predictions results/<model_name>/predictions.csv \
    --output results/<model_name>/evaluation.json
```

Metrics computed:
- **ROUGE-1, ROUGE-2, ROUGE-L** (F1 scores)
- **BERTScore** (F1, using `microsoft/deberta-xlarge-mnli`)

Results stored in `results/<model_name>/evaluation.json`.

---

## 10. Benchmark Results

All models evaluated on the **NewsSumm test set** under identical data splits:

| Model | Params (M) | Training Type | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|-------|-----------|---------------|---------|---------|---------|-----------|
| PRIMERA | ~150 | From scratch | 0.3818 | 0.1928 | 0.2833 | 0.8147 |
| LongT5-base | ~248 | From scratch | 0.2658 | 0.1162 | 0.2055 | 0.7806 |
| LED-base | ~162 | From scratch | 0.3686 | 0.1862 | 0.2762 | 0.8153 |
| Flan-T5-XL | ~3000 | From scratch | 0.2349 | 0.1279 | 0.1833 | 0.7853 |
| Flan-T5-XXL | ~11000 | PEFT | 0.2760 | 0.1620 | 0.2170 | 0.7990 |
| Mistral-7B-Instruct | ~7000 | PEFT | 0.3952 | 0.2213 | 0.3059 | 0.8834 |
| Llama-3-8B-Instruct | ~8000 | PEFT | 0.2854 | 0.1862 | 0.3892 | 0.8680 |
| Qwen2-7B-Instruct | ~7000 | PEFT | 0.2734 | 0.1341 | 0.3840 | 0.8720 |
| Gemma-2-9B-Instruct | ~9000 | PEFT | 0.2833 | 0.1766 | 0.4013 | 0.8850 |
| Mixtral-8x7B-Instruct | ~47000 | PEFT | 0.1787 | 0.1622 | 0.3770 | 0.8792 |
| **HiGS (Ours)** | **~250** | **From scratch** | **0.2443** | **0.1123** | **0.2122** | **0.8466** |

> **Key Finding:** Despite training under significant hardware and time constraints, HiGS achieves a BERTScore of 0.8466 and ROUGE-L of 0.2122 with only ~250M parameters. While larger instruction-tuned LLMs (Gemma-2-9B, Mistral-7B) achieve higher absolute scores, HiGS demonstrates that explicit graph-based modeling of inter-document structure provides a highly parameter-efficient alternative — using **28–188× fewer parameters** than the LLM baselines.

> **⚠️ Important Note on Training Convergence:** Due to time and hardware/GPU constraints (1× NVIDIA T4 16GB + 1× RTX 3080 10GB), HiGS training could not be fully converged. Phase 1 training ran for 20 epochs with loss still trending downward, and Phase 2 decoder fine-tuning ran for 2 epochs. With additional training on more powerful hardware, we expect significant performance improvements. A sample evaluation notebook is available at [`notebooks/evaluate_higs_sample.ipynb`](notebooks/evaluate_higs_sample.ipynb).

---

## 11. Reproducing a Past Experiment

```bash
python scripts/train_baseline.py --config results/<run_name>/config.yaml
```

This ensures same hyperparameters, seed, and deterministic pipeline behavior.

---

## 12. Running on a New System

```bash
# Step 1: Clone
git clone https://github.com/Sahilnegi4444/HiGS_GAT_based_summarization.git
cd HiGS_Multi_document_abstract_summarization_in_Indian_English

# Step 2: Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Step 3: Download data & model
Download cleaned dataset from: https://drive.google.com/drive/folders/1l_5WC5gacZAnCjZCgcSC6ZvNc4Sa2Igc
Place in: data/processed/newssumm_cleaned.parquet

Download model checkpoint from: https://drive.google.com/drive/folders/1hqYPvjdl443WFcgfs9OA-73p0U5Nusbm
Place in: results/higs/best_checkpoint.pt

# Step 4: Preprocess (splits)
python scripts/preprocess.py --input data/processed/newssumm_cleaned.parquet --output data/processed

# Step 5: Train any model
python scripts/train_higs.py --config configs/higs.yaml --phase 1

# Step 6: Evaluate
python scripts/evaluate.py --predictions results/higs/predictions.csv --output results/higs/evaluation.json
```

---

## 13. Experiment Strategy

**Heavy GPU Training:**
- PRIMERA, LED, LongT5, HiGS (Novel)

**PEFT / LoRA Fine-tuning:**
- Flan-T5-XL/XXL, Mistral-7B, LLaMA-3-8B, Qwen2-7B, Gemma-2-9B, Mixtral-8x7B

---

## 14. Research Goals

This repository supports:
- Long-context multi-document summarization for Indian English news
- Hierarchical graph-based architectures for cross-document reasoning
- Systematic comparison of encoder-decoder vs. LLM-based approaches
- Parameter-efficient fine-tuning (LoRA/QLoRA) for large models
- Fully reproducible experiments with config-driven training

---

## Acknowledgment

This research was conducted under the Internship Program at **Suvidha Foundation (Suvidha Mahila Mandal)**, Nagpur, India. The author gratefully acknowledges the Research Mentorship Support provided throughout the development of this project.

---

## Citation

If you use this work, please cite:

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
