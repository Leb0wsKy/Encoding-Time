import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def train_model(model, data_fn, vocab_size, epochs=50, task_type='seq2seq'):
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        inputs, targets = data_fn()
        outputs = model(inputs)

        if task_type == 'classification':
            logits = outputs[:, -1, :]  # last token output for classification
            loss = loss_fn(logits, targets)
            losses.append(loss.item())
        else:  # 'seq2seq'
            loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
            losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    return losses