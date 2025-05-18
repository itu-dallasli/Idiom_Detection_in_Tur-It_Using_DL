from typing import Optional, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multi‑class focal loss ‑ https://arxiv.org/abs/1708.02002"""
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def forward(self, logits, target):
        ce_loss = self.ce(logits.view(-1, logits.size(-1)), target.view(-1))
        pt = torch.exp(-ce_loss)                      # prob of correct class
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


class EnhancedBertForIdiomDetection(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        use_bilstm: bool = True,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.1,
        use_pos_embeddings: bool = False,
        pos_vocab_size: int = 64,
        pos_emb_dim: int = 32,
        dropout_prob: float = 0.1,
        freeze_bert_layers: int = 0,
        use_crf: bool = True,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.use_bilstm = use_bilstm
        self.use_pos_embeddings = use_pos_embeddings
        self.use_crf = use_crf

        # ------------------------------------------------------------------
        # 1) Backbone transformer
        # ------------------------------------------------------------------
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path, add_pooling_layer=False)

        # freeze bottom encoder layers if requested
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_dim = self.config.hidden_size

        # ------------------------------------------------------------------
        # 2) Optional POS / morphology embedding
        # ------------------------------------------------------------------
        if self.use_pos_embeddings:
            self.pos_embeddings = nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0)
            hidden_dim += pos_emb_dim

        # ------------------------------------------------------------------
        # 3) Optional BiLSTM
        # ------------------------------------------------------------------
        if self.use_bilstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            hidden_dim = lstm_hidden_size * 2  # because bidirectional

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # ------------------------------------------------------------------
        # 4) CRF layer or focal‑loss softmax
        # ------------------------------------------------------------------
        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.focal_loss = FocalLoss(gamma=focal_gamma, ignore_index=-100)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        # ------------------ backbone ------------------
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state        # (B, T, H)

        # add POS / morph embeddings
        if self.use_pos_embeddings and pos_ids is not None:
            pos_vec = self.pos_embeddings(pos_ids)          # (B, T, D_pos)
            sequence_output = torch.cat([sequence_output, pos_vec], dim=-1)

        # optional BiLSTM
        if self.use_bilstm:
            sequence_output, _ = self.lstm(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)        # (B, T, num_labels)

        loss = None
        # ------------------ loss ------------------
        if labels is not None:
            if self.use_crf:
                # CRF expects mask=1 where tokens are valid (not padding)
                loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            else:
                loss = self.focal_loss(emissions, labels)

        # ------------------ decode ------------------
        if self.use_crf:
            # list[list[int]] with variable lengths
            decode_out = self.crf.decode(emissions, mask=attention_mask.bool())
            # pad back to tensor for compatibility (fill with -100)
            max_len = input_ids.size(1)
            decoded = torch.full(labels.shape if labels is not None else (len(decode_out), max_len),
                                 fill_value=-100, dtype=torch.long, device=input_ids.device)
            for i, seq in enumerate(decode_out):
                decoded[i, :len(seq)] = torch.tensor(seq, device=input_ids.device)
        else:
            decoded = emissions.argmax(dim=-1)              # (B, T)

        return {
            "loss": loss,
            "logits": decoded,              # already best‑path indices
            "hidden_states": bert_out.hidden_states if self.config.output_hidden_states else None
        }
