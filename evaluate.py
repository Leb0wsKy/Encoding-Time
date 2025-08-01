import matplotlib.pyplot as plt

def plot_loss(losses_list, title="Loss"):
    model_labels = ["LSTM", "RNN", "Transformer"]
    
    plt.figure(figsize=(10, 6))
    
    for losses, label in zip(losses_list, model_labels):
        plt.plot(losses, label=label)
    
    plt.title(f"{title.upper()} Task - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Results/{title}_loss_plot.png")
    # Display without blocking script
    plt.show(block=False)
    plt.pause(6)
    plt.close()
