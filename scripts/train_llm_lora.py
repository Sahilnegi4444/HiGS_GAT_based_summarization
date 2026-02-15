"""
LLM LoRA/QLoRA Fine-tuning Script

Usage:
    python scripts/train_llm_lora.py \
        --config configs/llm_lora.yaml \
        --model mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pandas as pd
import yaml
from tqdm.auto import tqdm
from transformers import TrainingArguments, Trainer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.llm_lora import load_llm_with_lora, LLMSummarizationDataset


def main():
    parser = argparse.ArgumentParser(description="Train LLM with LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    model_short = args.model.split("/")[-1]
    run_name = f"llm_{model_short}"
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config["model_id"] = args.model
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load model with LoRA
    model, tokenizer = load_llm_with_lora(
        model_name=args.model,
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        use_4bit=config.get("use_4bit", True),
    )

    # Load data
    data_dir = Path(config.get("data_dir", "data/processed"))
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")

    max_samples = config.get("max_train_samples", len(train_df))
    train_df = train_df.head(max_samples)

    train_ds = LLMSummarizationDataset(
        train_df["articles_clean"].tolist(),
        train_df["summary_clean"].tolist(),
        tokenizer,
        max_length=config.get("max_length", 2048),
    )
    val_ds = LLMSummarizationDataset(
        val_df["articles_clean"].tolist()[:1000],
        val_df["summary_clean"].tolist()[:1000],
        tokenizer,
        max_length=config.get("max_length", 2048),
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(results_dir / "checkpoints"),
        num_train_epochs=config.get("num_epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 1),
        per_device_eval_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        warmup_steps=config.get("warmup_steps", 100),
        logging_steps=50,
        eval_steps=config.get("eval_steps", 500),
        save_steps=config.get("save_steps", 500),
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True,
        report_to="none",
        seed=config.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Save final model
    model.save_pretrained(results_dir / "final_model")
    tokenizer.save_pretrained(results_dir / "final_model")

    meta = {
        "model_id": args.model,
        "training_time_sec": elapsed,
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "use_4bit": config.get("use_4bit", True),
    }
    with open(results_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Results: {results_dir}")


if __name__ == "__main__":
    main()
