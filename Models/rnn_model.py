import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_classes=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # For copying/matching
        self.classifier = nn.Linear(hidden_size, num_classes)  # For sentiment ordering

    def forward(self, x, task="copying"):
        x = self.embed(x)
        out, _ = self.rnn(x)
        if task == "sentiment":
            # Pool last hidden state for classification
            out = self.classifier(out[:, -1, :])
        else:
            out = self.fc(out)
        return out
