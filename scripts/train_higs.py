"""
HiGS Two-Phase Training Pipeline (Dual-Encoder Fusion Architecture)
====================================================================

Architecture:
  Pathway A: BERT Encoder → GAT Layers → MLP Projection (R^512 → R^768)
  Pathway B: Native BART Encoder (frozen, R^768)
  Fusion:    torch.cat([PathwayB, PathwayA]) → BART Decoder

Two-Phase Strategy:
  Phase 1 — Train the Graph (Decoder frozen)
    - BERT encoder + GAT layers + projection are trained
    - BART encoder and decoder are frozen
    - Goal: Learn structural graph routing

  Phase 2 — Align the Decoder (Encoder/GAT frozen)
    - BERT encoder + GAT layers are frozen
    - BART decoder cross-attention layers are unfrozen
    - Goal: Align decoder with graph-enriched representations

Usage:
    # Phase 1
    python scripts/train_higs.py --config configs/higs.yaml --phase 1

    # Phase 2 (resume from Phase 1 best checkpoint)
    python scripts/train_higs.py --config configs/higs.yaml --phase 2 \
        --checkpoint results/higs_phase1/best_checkpoint.pt

    # Resume training from any checkpoint
    python scripts/train_higs.py --config configs/higs.yaml --phase 2 \
        --checkpoint results/higs_phase2/checkpoint_step10000.pt --resume
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
from transformers import AutoTokenizer, BartTokenizer, get_cosine_schedule_with_warmup
import pandas as pd
import yaml
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.higs_model import HiGraphSum, GraphSumDataset


# ── CHECKPOINT KEY REMAPPING ────────────────────────────────────────
def remap_checkpoint_keys(state_dict):
    """Remap old-format checkpoint keys to the current Dual-Encoder architecture."""
    new_sd = {}
    for key, value in state_dict.items():
        if key == "projection.weight":
            new_sd["projection.0.weight"] = value
        elif key == "projection.bias":
            new_sd["projection.0.bias"] = value
        elif key in ("residual_gate",) or "word_projection" in key:
            continue  # skip removed layers from old architecture
        else:
            new_sd[key] = value
    return new_sd


def load_checkpoint(model, checkpoint_path, device, phase, resume=False):
    """
    Load checkpoint with architecture-aware key handling.

    For Phase 2 loading a Phase 1 checkpoint:
      - Loads encoder + GAT + projection weights
      - Skips corrupted decoder weights (decoder was frozen in Phase 1,
        so its weights are untrained noise)
      - Keeps the fresh pretrained BART decoder weights

    For resuming within the same phase:
      - Loads all weights directly
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = remap_checkpoint_keys(ckpt.get("model_state_dict", ckpt))

    if not resume and phase == 2 and "phase1" in str(checkpoint_path).lower():
        # Phase 2 loading Phase 1: skip decoder keys to preserve pretrained BART
        filtered = {k: v for k, v in state_dict.items()
                    if not k.startswith("decoder.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"  ✅ Loaded Phase 1 weights (decoder keys preserved from pretrained BART)")
        print(f"     Skipped {len(state_dict) - len(filtered)} decoder keys")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Loaded all weights")

    if missing:
        print(f"  ⚠️  Missing keys: {len(missing)} (expected for new layers)")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {len(unexpected)}")

    return ckpt


def freeze_for_phase(model, phase):
    """Apply the correct freeze/unfreeze strategy for the given phase."""
    if phase == 1:
        # Phase 1: Train BERT + GAT + Projection. Freeze BART entirely.
        print("  ❄️  Phase 1: Freezing BART encoder and decoder")
        for param in model.decoder.parameters():
            param.requires_grad = False
        # BERT encoder is trainable
        for param in model.sentence_encoder.parameters():
            param.requires_grad = True
        # GAT layers are trainable
        for param in model.gat_layers.parameters():
            param.requires_grad = True
        # Projection is trainable
        for param in model.projection.parameters():
            param.requires_grad = True

    elif phase == 2:
        # Phase 2: Freeze BERT + GAT. Unfreeze decoder cross-attention.
        print("  ❄️  Phase 2: Freezing BERT encoder and GAT layers")
        for param in model.sentence_encoder.parameters():
            param.requires_grad = False
        for param in model.gat_layers.parameters():
            param.requires_grad = False

        # Freeze projection (already trained in Phase 1)
        for param in model.projection.parameters():
            param.requires_grad = False

        # Unfreeze BART decoder (or just cross-attention for fine control)
        print("  🔓 Phase 2: Unfreezing BART decoder")
        for param in model.decoder.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total "
          f"({100*trainable/total:.1f}%)")
    return trainable, total


# ── TRAINING LOOP ───────────────────────────────────────────────────
def train(config, phase, checkpoint_path=None, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  HiGS Training — Phase {phase}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")

    # ── Paths ──
    run_name = config.get("run_name", f"higs_phase{phase}")
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump({"phase": phase, **config}, f)

    # ── Data ──
    data_dir = Path(config.get("data_dir", "data/processed"))
    print(f"📖 Loading data from {data_dir}...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Tokenizers ──
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bart_tok = BartTokenizer.from_pretrained("facebook/bart-base")

    # ── Datasets ──
    max_sents = config.get("max_sents", 30)
    max_sent_len = config.get("max_sent_len", 64)
    max_summary_len = config.get("max_summary_len", 128)

    train_ds = GraphSumDataset(
        train_df["articles_clean"].tolist(),
        train_df["summary_clean"].tolist(),
        bert_tok, bart_tok, max_sents, max_sent_len, max_summary_len,
    )
    val_ds = GraphSumDataset(
        val_df["articles_clean"].tolist(),
        val_df["summary_clean"].tolist(),
        bert_tok, bart_tok, max_sents, max_sent_len, max_summary_len,
    )

    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("gradient_accumulation_steps", 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── Model ──
    print("\n📦 Initializing HiGS model...")
    model = HiGraphSum(
        num_gat_layers=config.get("num_gat_layers", 2),
        gat_hidden_dim=config.get("gat_hidden_dim", 512),
        dropout=config.get("dropout", 0.2),
        label_smoothing=config.get("label_smoothing", 0.1),
    ).to(device)

    # ── Load checkpoint ──
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if checkpoint_path:
        ckpt = load_checkpoint(model, checkpoint_path, device, phase, resume)
        if resume:
            start_epoch = ckpt.get("epoch", 0)
            if isinstance(start_epoch, float):
                start_epoch = int(start_epoch)
            global_step = ckpt.get("step", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"  Resuming from epoch {start_epoch}, step {global_step}")

    # ── Freeze strategy ──
    trainable, total = freeze_for_phase(model, phase)

    # ── Optimizer & Scheduler ──
    lr = config.get(f"phase{phase}_lr", 3e-5)
    epochs = config.get(f"phase{phase}_epochs", 2)
    warmup_steps = config.get("warmup_steps", 200)
    weight_decay = config.get("weight_decay", 0.01)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = max(1, (len(train_loader) * epochs) // grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    save_steps = config.get("save_steps", 5000)

    print(f"\n🏋️ Training Configuration:")
    print(f"   Epochs: {epochs} | Batch: {batch_size} × {grad_accum} accum")
    print(f"   LR: {lr} | Warmup: {warmup_steps} | Total steps: {total_steps}")
    print(f"   Save every {save_steps} steps")

    # ── Training ──
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs}")
        for step_i, batch in enumerate(pbar):
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            try:
                with autocast(enabled=(device.type == "cuda")):
                    loss = model(batch_dev)
                    loss = loss / grad_accum

                scaler.scale(loss).backward()

                if (step_i + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.get("max_grad_norm", 1.0)
                    )
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()

                    # Only step scheduler if optimizer actually stepped
                    if scale_after >= scale_before:
                        scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * grad_accum
                num_batches += 1
                pbar.set_postfix(
                    loss=f"{epoch_loss/num_batches:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                # Periodic checkpoint
                if global_step > 0 and global_step % save_steps == 0:
                    ckpt_path = results_dir / f"checkpoint_step{global_step}.pt"
                    torch.save({
                        "epoch": epoch + (step_i / len(train_loader)),
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "train_loss": epoch_loss / num_batches,
                    }, ckpt_path)
                    print(f"\n  💾 Checkpoint saved: {ckpt_path.name}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  ⚠️ OOM at step {step_i}, skipping batch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise e

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # ── Validation ──
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                try:
                    loss = model(batch_dev)
                    val_loss += loss.item()
                    val_batches += 1
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"\n📊 Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = results_dir / "best_checkpoint.pt"
            torch.save({
                "epoch": epoch + 1,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, best_path)
            print(f"  ✅ Best checkpoint saved (val_loss={avg_val_loss:.4f})")

        # Save epoch checkpoint
        epoch_path = results_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, epoch_path)

    elapsed = time.time() - start_time

    # ── Save metadata ──
    meta = {
        "phase": phase,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
        "training_time_sec": elapsed,
        "training_time_min": round(elapsed / 60, 1),
        "device": str(device),
        "trainable_params": trainable,
        "total_params": total,
        "learning_rate": lr,
    }
    with open(results_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Phase {phase} complete in {elapsed/60:.1f} min")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Total steps:   {global_step:,}")
    print(f"   Results dir:   {results_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train HiGS model (Dual-Encoder Fusion Architecture)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="Training phase (1=train graph, 2=align decoder)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to load (required for Phase 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training (load optimizer state, epoch, step)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train(config, args.phase, args.checkpoint, args.resume)


if __name__ == "__main__":
    main()
