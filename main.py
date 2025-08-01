import torch
from Models.lstm_model import LSTMModel
from Models.rnn_model import RNNModel
from Models.transformer_model import TransformerModel
from Datasets.copy_task import generate_copy_data
from Datasets.delayed_match import generate_delayed_match_data
from Datasets.pattern_completion import generate_pattern_completion_data
from train import train_model
from evaluate import plot_loss

# Optional: accuracy function for classification
def accuracy(preds, targets):
    pred_labels = torch.argmax(preds, dim=-1)
    return (pred_labels == targets).float().mean().item()

# === CONFIG ===
model_types = ["lstm", "rnn", "transformer"]
task_types = ["pattern completion","copying", "matching"]
vocab_size = 20
hidden_size = 128
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === TASK SETUP ===
for task_type in task_types:
    if task_type == "copying":
        def data_fn():
            x, y = generate_copy_data(seq_len=100, vocab_size=vocab_size, batch_size=32)
            return x.to(device), y.to(device)
        task_mode = "seq2seq"

    elif task_type == "matching":
        def data_fn():
            x, y = generate_delayed_match_data(delay=100, vocab_size=vocab_size, batch_size=32)
            return x.to(device), y.to(device)
        task_mode = "seq2seq"

    elif task_type == "pattern completion":
        def data_fn():
            x, y = generate_pattern_completion_data(seq_len=6, vocab_size=5, batch_size=32)
            return x.to(device), y.to(device)
        task_mode = "classification"
    task_losses = []
    # === MODEL ===
    for model_type in model_types:
        if model_type == "lstm":
            model = LSTMModel(vocab_size)
        elif model_type == "rnn":
            model = RNNModel(vocab_size)
        elif model_type == "transformer":
            model = TransformerModel(vocab_size)

        model.to(device)


        losses = train_model(model, data_fn, vocab_size, epochs, task_mode)
        task_losses.append(losses)
    plot_loss(task_losses, title=task_type)