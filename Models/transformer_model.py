import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, nhead=2, num_layers=1, num_classes=5, max_len=250):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)  # For copying/matching
        self.classifier = nn.Linear(hidden_size, num_classes)  # For sentiment ordering

    def forward(self, x, task="copying"):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer(x)
        if task == "sentiment":
            # Pool with mean for classification
            output = self.classifier(output.mean(dim=1))
        else:
            output = self.fc(output)
        return output
