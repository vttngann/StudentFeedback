import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERTMultiTask(nn.Module):

    def __init__(
        self,
        model_name="vinai/phobert-base",
        n_sentiment=3,
        n_topic=4,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_sentiment)
        )

        self.topic_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, n_topic)
        )

    def mean_pooling(self, last_hidden_state, attention_mask):

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (last_hidden_state * mask).sum(dim=1)

        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)

        return pooled

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        pooled = self.dropout(pooled)

        sentiment_logits = self.sentiment_head(pooled)

        topic_logits = self.topic_head(pooled)

        return sentiment_logits, topic_logits