"""
LLM LoRA/QLoRA Fine-tuning Wrapper

Supports decoder-only LLMs with parameter-efficient fine-tuning:
  - Mistral-7B-Instruct
  - LLaMA-3-8B-Instruct
  - Qwen2-7B-Instruct
  - Gemma-2-9B-Instruct
  - Mixtral-8x7B-Instruct
"""

import yaml
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_llm_with_lora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    target_modules: list[str] | None = None,
) -> tuple:
    """
    Load a decoder-only LLM with QLoRA configuration.

    Args:
        model_name: HuggingFace model ID
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: dropout for LoRA layers
        use_4bit: whether to use 4-bit quantization (QLoRA)
        target_modules: list of modules to apply LoRA to

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading LLM: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Default target modules for common architectures
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# Prompt Template
# ============================================================================

SUMMARIZATION_PROMPT = (
    "You are a news editor. Write a concise, factual summary of the following "
    "Indian news articles. Focus on key events, entities, and outcomes. "
    "Do not hallucinate or add information not present in the articles.\n\n"
    "Articles:\n{articles}\n\n"
    "Summary:"
)


def format_prompt(article: str) -> str:
    """Format an article into a summarization prompt."""
    return SUMMARIZATION_PROMPT.format(articles=article)


# ============================================================================
# Dataset for LLM Fine-tuning
# ============================================================================

class LLMSummarizationDataset(Dataset):
    """
    Dataset for supervised fine-tuning of decoder-only LLMs.
    Creates prompt + completion pairs for causal LM training.
    """

    def __init__(
        self,
        articles: list[str],
        summaries: list[str],
        tokenizer,
        max_length: int = 2048,
    ):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> dict:
        prompt = format_prompt(self.articles[idx])
        completion = self.summaries[idx]
        full_text = prompt + " " + completion + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Mask prompt tokens from loss computation
        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length
        )
        prompt_len = len(prompt_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
