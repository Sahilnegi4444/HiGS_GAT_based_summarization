"""
Baseline Model Loader

Config-driven loader for encoder-decoder baseline models:
  - PRIMERA (allenai/PRIMERA)
  - LED (allenai/led-base-16384)
  - LongT5 (google/long-t5-tglobal-base)
  - Flan-T5-XL (google/flan-t5-xl)
"""

import yaml
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LEDForConditionalGeneration,
    LEDTokenizer,
)


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    "primera": "allenai/PRIMERA",
    "led": "allenai/led-base-16384",
    "longt5": "google/long-t5-tglobal-base",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-t5-xxl": "google/flan-t5-xxl",
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_baseline_model(config: dict) -> tuple:
    """
    Load a baseline model and tokenizer from config.

    Args:
        config: dict with at least 'model_name' and 'max_input_length'

    Returns:
        (model, tokenizer) tuple
    """
    model_name = config["model_name"]
    hf_name = MODEL_REGISTRY.get(model_name, model_name)

    print(f"Loading model: {hf_name}")

    if model_name == "led":
        tokenizer = LEDTokenizer.from_pretrained(hf_name)
        model = LEDForConditionalGeneration.from_pretrained(hf_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)

    return model, tokenizer


# ============================================================================
# Dataset for Baselines
# ============================================================================

class BaselineSummarizationDataset(Dataset):
    """
    Simple dataset for encoder-decoder baselines.
    Concatenates multi-document input as a single sequence.
    """

    def __init__(
        self,
        articles: list[str],
        summaries: list[str],
        tokenizer,
        max_input_length: int = 4096,
        max_target_length: int = 128,
        doc_separator: str = " [DOC] ",
    ):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.doc_separator = doc_separator

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> dict:
        article = self.articles[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(
            article,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            summary,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }
