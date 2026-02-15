"""
HiGS Two-Phase Training Script

Usage:
    # Phase 1: Full training (encoder + GAT + decoder)
    python scripts/train_higs.py --config configs/higs.yaml --phase 1

    # Phase 2: Decoder-only fine-tuning (encoder/GAT frozen)
    python scripts/train_higs.py --config configs/higs.yaml --phase 2 \
        --checkpoint results/higs_phase1/best_checkpoint.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, BartTokenizer
import pandas as pd
import yaml
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.higs_model import HiGraphSum, GraphSumDataset


def train(config: dict, phase: int, checkpoint_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase: {phase}")

    # Paths
    run_name = config.get("run_name", f"higs_phase{phase}")
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump({"phase": phase, **config}, f)

    # Load data
    data_dir = Path(config.get("data_dir", "data/processed"))
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")

    # Tokenizers
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")

    # Datasets
    train_ds = GraphSumDataset(
        train_df["articles_clean"].tolist(),
        train_df["summary_clean"].tolist(),
        bert_tok, bart_tok,
        max_sents=config.get("max_sents", 30),
        max_sent_len=config.get("max_sent_len", 64),
        max_summary_len=config.get("max_summary_len", 128),
    )
    val_ds = GraphSumDataset(
        val_df["articles_clean"].tolist(),
        val_df["summary_clean"].tolist(),
        bert_tok, bart_tok,
        max_sents=config.get("max_sents", 30),
        max_sent_len=config.get("max_sent_len", 64),
        max_summary_len=config.get("max_summary_len", 128),
    )

    batch_size = config.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = HiGraphSum(
        num_gat_layers=config.get("num_gat_layers", 2),
        gat_hidden_dim=config.get("gat_hidden_dim", 512),
        dropout=config.get("dropout", 0.2),
        label_smoothing=config.get("label_smoothing", 0.1),
    ).to(device)

    # Load checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"   Loaded from epoch {ckpt.get('epoch', '?')}")

    # Phase 2: Freeze encoder + GAT
    if phase == 2:
        print("❄️  Freezing encoder and GAT layers")
        for param in model.sentence_encoder.parameters():
            param.requires_grad = False
        for param in model.gat_layers.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Optimizer
    lr = config.get(f"phase{phase}_lr", config.get("learning_rate", 3e-5))
    epochs = config.get(f"phase{phase}_epochs", config.get("num_epochs", 2))

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Training
    best_val_loss = float("inf")
    global_step = 0
    save_steps = config.get("save_steps", 5000)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(enabled=torch.cuda.is_available()):
                loss = model(batch_dev)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("max_grad_norm", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            # Periodic save
            if global_step % save_steps == 0:
                torch.save({
                    "epoch": epoch + (global_step / len(train_loader)),
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "train_loss": total_loss / (global_step % len(train_loader) or 1),
                }, results_dir / f"checkpoint_step{global_step}.pt")

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                loss = model(batch_dev)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch + 1,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "train_loss": avg_loss,
                "val_loss": avg_val,
            }, results_dir / "best_checkpoint.pt")
            print(f"   ✓ Best checkpoint saved (val_loss={avg_val:.4f})")

    elapsed = time.time() - start_time

    meta = {
        "phase": phase,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
        "training_time_sec": elapsed,
        "device": str(device),
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(results_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Phase {phase} complete in {elapsed/60:.1f} min")
    print(f"   Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train HiGS model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train(config, args.phase, args.checkpoint)


if __name__ == "__main__":
    main()
