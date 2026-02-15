"""
Config-Driven Baseline Training

Usage:
    python scripts/train_baseline.py --config configs/led_baseline.yaml

Supports: PRIMERA, LED, LongT5, Flan-T5-XL, Flan-T5-XXL
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import yaml
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baselines import load_baseline_model, BaselineSummarizationDataset


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    run_name = config.get("run_name", config["model_name"])
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load data
    data_dir = Path(config.get("data_dir", "data/processed"))
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")

    # Load model
    model, tokenizer = load_baseline_model(config)
    model.to(device)

    # Datasets
    train_ds = BaselineSummarizationDataset(
        train_df["articles_clean"].tolist(),
        train_df["summary_clean"].tolist(),
        tokenizer,
        max_input_length=config.get("max_input_length", 4096),
        max_target_length=config.get("max_target_length", 128),
    )
    val_ds = BaselineSummarizationDataset(
        val_df["articles_clean"].tolist(),
        val_df["summary_clean"].tolist(),
        tokenizer,
        max_input_length=config.get("max_input_length", 4096),
        max_target_length=config.get("max_target_length", 128),
    )

    batch_size = config.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Optimizer
    lr = config.get("learning_rate", 3e-5)
    epochs = config.get("num_epochs", 3)
    warmup = config.get("warmup_steps", 500)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    # Training loop
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "train_loss": avg_loss,
                "val_loss": avg_val,
            }, results_dir / "best_checkpoint.pt")
            print(f"   ✓ Saved best checkpoint (val_loss={avg_val:.4f})")

    elapsed = time.time() - start_time

    # Save metadata
    meta = {
        "model_name": config["model_name"],
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "training_time_sec": elapsed,
        "device": str(device),
        "seed": config.get("seed", 42),
    }
    with open(results_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Training complete in {elapsed/60:.1f} min")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Results: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train(config)


if __name__ == "__main__":
    main()
