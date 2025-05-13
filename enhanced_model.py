import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class EnhancedBertForIdiomDetection(nn.Module):
    def __init__(self,
                 model_name="bert-base-multilingual-cased",
                 num_labels=5,  # Changed from 3 to 5 for BIOES
                 lstm_hidden_size=384,
                 lstm_layers=2,
                 lstm_dropout=0.3,
                 hidden_dropout=0.3,
                 use_layer_norm=True,
                 freeze_bert_layers=0):
        super(EnhancedBertForIdiomDetection, self).__init__()

        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Freeze specified number of BERT layers if needed
        if freeze_bert_layers > 0:
            modules = [self.bert.embeddings]
            modules.extend(self.bert.encoder.layer[:freeze_bert_layers])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        # Add a BiLSTM layer to capture context
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )

        # Classification layers for token-level BIOES tagging
        self.dropout = nn.Dropout(hidden_dropout)
        self.dense = nn.Linear(lstm_hidden_size*2, lstm_hidden_size)
        self.activation = nn.ReLU()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(lstm_hidden_size)
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)

        # Sentence-level classification head
        self.sentence_pooler = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(lstm_hidden_size, 2)  # Binary classification: idiomatic vs literal
        )

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

        # Loss weights for rebalancing
        self.token_loss_weight = 0.7
        self.sentence_loss_weight = 0.3

    def forward(self, input_ids, attention_mask, labels=None, sentence_labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get token-level representations
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Apply BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, 2*hidden_size]

        # Token-level classification
        x = self.dropout(lstm_output)
        x = self.dense(x)
        x = self.activation(x)
        if self.use_layer_norm:
            x = self.norm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)  # [batch_size, seq_len, num_labels]

        # Sentence-level classification
        # Use attention-weighted pooling
        attention_weights = torch.softmax(torch.matmul(x, x.transpose(-1, -2)), dim=-1)
        sentence_representation = torch.matmul(attention_weights, x).mean(dim=1)
        sentence_logits = self.sentence_pooler(sentence_representation)

        loss = None
        if labels is not None:
            # Create mask for CRF
            crf_mask = attention_mask.bool()

            # CRF loss (negative log-likelihood)
            token_loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')

            # Sentence-level classification loss
            if sentence_labels is not None:
                sentence_loss = nn.CrossEntropyLoss()(sentence_logits, sentence_labels)
                # Combine losses with weights
                loss = (self.token_loss_weight * token_loss + 
                       self.sentence_loss_weight * sentence_loss)
            else:
                loss = token_loss

        # CRF decoding for predictions
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        # Convert list of lists to tensor with padding
        max_len = emissions.size(1)
        pred_tensor = torch.zeros_like(input_ids)
        for i, pred_seq in enumerate(predictions):
            pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)

        return {
            'loss': loss,
            'logits': emissions,
            'predictions': pred_tensor,
            'sentence_logits': sentence_logits
        }

def convert_bio_to_bioes(bio_tags):
    """
    Convert BIO tags to BIOES tags
    B -> B (beginning)
    I -> I (inside)
    O -> O (outside)
    I -> E (end) if next tag is not I
    """
    bioes_tags = bio_tags.copy()
    for i in range(len(bio_tags)):
        if bio_tags[i] == 2:  # I tag
            if i == len(bio_tags) - 1 or bio_tags[i + 1] != 2:  # Last token or next is not I
                bioes_tags[i] = 4  # E tag
    return bioes_tags

def convert_bioes_to_bio(bioes_tags):
    """
    Convert BIOES tags back to BIO tags
    B -> B (beginning)
    I -> I (inside)
    E -> I (end)
    O -> O (outside)
    S -> B (single)
    """
    bio_tags = bioes_tags.copy()
    for i in range(len(bioes_tags)):
        if bioes_tags[i] == 4:  # E tag
            bio_tags[i] = 2  # Convert to I
        elif bioes_tags[i] == 3:  # S tag
            bio_tags[i] = 1  # Convert to B
    return bio_tags 