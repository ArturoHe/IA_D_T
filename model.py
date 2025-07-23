# model.py
import torch.nn as nn


class SimpleDecisionTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32, seq_len=20, num_classes=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4), num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        x = self.embed(x)  # -> (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # -> (seq_len, batch, embed_dim)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # -> (batch, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # -> (batch, embed_dim)
        return self.head(x)  # -> (batch, num_classes)
