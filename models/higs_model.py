"""
HiGS: Hierarchical Graph-Based Summarization Model

Architecture:
  1. BERT Encoder — sentence-level representations
  2. GAT Layers  — graph-based cross-document reasoning
  3. BART Decoder — conditional summary generation

Components:
  - GraphAttentionLayer: Single-head graph attention
  - HiGraphSum: Full model with encode + generate
  - GraphSumDataset: Dataset class with dual tokenizers (BERT + BART)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput
import spacy

nlp = spacy.load("en_core_web_sm")


# ============================================================================
# NLP Utilities
# ============================================================================

def split_into_sentences(text: str, min_len: int = 10) -> list[str]:
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > min_len]


def extract_entities(sentence: str) -> set[str]:
    """Extract named entities (PERSON, ORG, GPE, LOC) from a sentence."""
    if not sentence:
        return set()
    doc = nlp(sentence)
    return {ent.text.lower() for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}}


def build_adjacency_matrix(
    sentences: list[str],
    sent_embeddings: torch.Tensor,
    threshold: float = 0.75,
) -> torch.Tensor:
    """
    Build a binary adjacency matrix based on:
      - Cosine similarity (> threshold)
      - Named entity overlap between sentence pairs
    """
    valid_indices = [i for i, s in enumerate(sentences) if s.strip()]
    num_valid = len(valid_indices)
    num_total = sent_embeddings.size(0)
    adj = torch.zeros(num_total, num_total)

    if num_valid > 1:
        valid_emb = sent_embeddings[valid_indices]
        normed = F.normalize(valid_emb, p=2, dim=1)
        sim = torch.mm(normed, normed.t())

        for i_idx, i in enumerate(valid_indices):
            for j_idx, j in enumerate(valid_indices):
                if sim[i_idx, j_idx] > threshold:
                    adj[i, j] = 1.0

        entities = [extract_entities(sentences[i]) for i in valid_indices]
        for i_idx, i in enumerate(valid_indices):
            for j_idx, j in enumerate(valid_indices):
                if i_idx < j_idx and (entities[i_idx] & entities[j_idx]):
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

    adj.fill_diagonal_(0)
    return adj


# ============================================================================
# Graph Attention Layer
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   (batch, num_nodes, in_features)
            adj: (batch, num_nodes, num_nodes)
        Returns:
            h_out: (batch, num_nodes, out_features)
        """
        Wh = self.W(h)
        B, N, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))

        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))
        att = F.softmax(e, dim=-1).masked_fill(mask, 0.0)
        att = self.dropout(att)

        return torch.bmm(att, Wh)


# ============================================================================
# HiGraphSum Model
# ============================================================================

class HiGraphSum(nn.Module):
    """
    Hierarchical Graph-based Summarization model.

    Pipeline:
      sentence tokens ──> BERT CLS ──> GAT layers ──> projection ──> BART decoder
    """

    def __init__(
        self,
        num_gat_layers: int = 2,
        gat_hidden_dim: int = 512,
        dropout: float = 0.2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        # Encoder
        self.sentence_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.bert_hidden_dim = self.sentence_encoder.config.hidden_size

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                self.bert_hidden_dim if i == 0 else gat_hidden_dim,
                gat_hidden_dim,
                dropout,
            )
            for i in range(num_gat_layers)
        ])
        self.gat_dropout = nn.Dropout(dropout)

        # Decoder
        self.decoder = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        bart_hidden_dim = self.decoder.config.d_model
        self.projection = nn.Linear(gat_hidden_dim, bart_hidden_dim)

        # Tokenizer (for adjacency matrix building)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Loss
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=-100
        )

    def encode_sentences(
        self,
        sent_input_ids: torch.Tensor,
        sent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each sentence independently with BERT and extract [CLS] embeddings.

        Args:
            sent_input_ids:      (batch, num_sents, max_sent_len)
            sent_attention_mask: (batch, num_sents, max_sent_len)
        Returns:
            sent_embeddings: (batch, num_sents, bert_hidden_dim)
        """
        B, S, L = sent_input_ids.size()
        flat_ids = sent_input_ids.view(-1, L)
        flat_mask = sent_attention_mask.view(-1, L)
        outputs = self.sentence_encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return cls.view(B, S, -1)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass with loss computation.

        Args:
            batch: dict with keys:
                sent_input_ids, sent_attention_mask, sentences_raw,
                summary_input_ids, summary_attention_mask
        Returns:
            loss: scalar tensor
        """
        sent_ids = batch["sent_input_ids"]
        sent_mask = batch["sent_attention_mask"]
        raw = batch["sentences_raw"]
        B = sent_ids.size(0)

        # Encode sentences
        node_features = self.encode_sentences(sent_ids, sent_mask)

        # Build adjacency matrices
        adj_list = []
        for i in range(B):
            sents = raw[i].split("|||") if isinstance(raw[i], str) else list(raw[i])
            adj_list.append(build_adjacency_matrix(sents, node_features[i], 0.75))
        adj = torch.stack(adj_list).to(node_features.device)

        # GAT layers
        h = node_features
        for gat in self.gat_layers:
            h = F.relu(gat(h, adj))
            h = self.gat_dropout(h)

        # Project to BART hidden dim
        encoder_hidden = self.projection(h)

        # Decoder
        labels = batch["summary_input_ids"].clone()
        labels[labels == self.decoder.config.pad_token_id] = -100

        outputs = self.decoder(
            encoder_outputs=(encoder_hidden,),
            attention_mask=torch.ones(B, h.size(1)).to(encoder_hidden.device),
            labels=labels,
            decoder_attention_mask=batch["summary_attention_mask"],
        )
        return outputs.loss

    @torch.no_grad()
    def generate_summary(
        self,
        batch: dict,
        num_beams: int = 4,
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Generate summary token IDs for a batch.

        Args:
            batch: dict with sent_input_ids, sent_attention_mask, sentences_raw
            num_beams: beam search width
            max_length: maximum output length
        Returns:
            generated_ids: (batch, seq_len) token IDs
        """
        sent_ids = batch["sent_input_ids"]
        sent_mask = batch["sent_attention_mask"]
        raw = batch["sentences_raw"]
        B = sent_ids.size(0)

        node_features = self.encode_sentences(sent_ids, sent_mask)

        adj_list = []
        for i in range(B):
            sents = raw[i].split("|||") if isinstance(raw[i], str) else list(raw[i])
            adj_list.append(build_adjacency_matrix(sents, node_features[i], 0.75))
        adj = torch.stack(adj_list).to(node_features.device)

        h = node_features
        for gat in self.gat_layers:
            h = F.relu(gat(h, adj))
            h = self.gat_dropout(h)

        encoder_hidden = self.projection(h)
        enc_mask = torch.ones(B, h.size(1)).to(encoder_hidden.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        return self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
        )


# ============================================================================
# Dataset
# ============================================================================

class GraphSumDataset(Dataset):
    """
    Dataset for HiGS model.

    Uses dual tokenizers:
      - BERT tokenizer for input sentences (encoder side)
      - BART tokenizer for summaries (decoder side)
    """

    def __init__(
        self,
        articles: list[str],
        summaries: list[str],
        bert_tokenizer,
        bart_tokenizer,
        max_sents: int = 30,
        max_sent_len: int = 64,
        max_summary_len: int = 128,
    ):
        self.articles = articles
        self.summaries = summaries
        self.bert_tokenizer = bert_tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.max_sents = max_sents
        self.max_sent_len = max_sent_len
        self.max_summary_len = max_summary_len

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> dict:
        article = self.articles[idx]
        summary = self.summaries[idx]
        sentences = split_into_sentences(article)[:self.max_sents]

        # Tokenize sentences with BERT
        if sentences:
            enc = self.bert_tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=self.max_sent_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
        else:
            input_ids = torch.zeros((0, self.max_sent_len), dtype=torch.long)
            attention_mask = torch.zeros((0, self.max_sent_len), dtype=torch.long)

        # Pad to max_sents
        n = input_ids.size(0)
        if n < self.max_sents:
            p = self.max_sents - n
            pad_ids = torch.full(
                (p, self.max_sent_len),
                self.bert_tokenizer.pad_token_id,
                dtype=torch.long,
            )
            input_ids = torch.cat([input_ids, pad_ids])
            attention_mask = torch.cat(
                [attention_mask, torch.zeros((p, self.max_sent_len), dtype=torch.long)]
            )

        # Raw sentences for adjacency matrix construction
        padded_sents = sentences + [""] * (self.max_sents - len(sentences))

        # Tokenize summary with BART (critical: different tokenizer from encoder)
        sum_enc = self.bart_tokenizer(
            summary,
            padding="max_length",
            truncation=True,
            max_length=self.max_summary_len,
            return_tensors="pt",
        )

        return {
            "sent_input_ids": input_ids,
            "sent_attention_mask": attention_mask,
            "sentences_raw": "|||".join(padded_sents),
            "summary_input_ids": sum_enc["input_ids"].squeeze(0),
            "summary_attention_mask": sum_enc["attention_mask"].squeeze(0),
            "reference": summary,
        }
