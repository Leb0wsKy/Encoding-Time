import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  
        self.classifier = nn.Linear(hidden_size, num_classes)  

    def forward(self, x, task="copying"):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        if task == "sentiment":
            # Pool last hidden state for classification
            output = self.classifier(output[:, -1, :])
        else:
            output = self.fc(output)
        return output
