import torch
import random

def generate_pattern_completion_data(seq_len=6, vocab_size=5, batch_size=32):
    input_seqs = []
    targets = []

    for _ in range(batch_size):
        # Choose a base pattern (length 2 or 3) randomly
        pattern_len = random.choice([2, 3])
        pattern = [random.randint(1, vocab_size - 1) for _ in range(pattern_len)]

        # Generate sequence by repeating the pattern until seq_len
        full_seq = []
        while len(full_seq) < seq_len + 1:
            full_seq.extend(pattern)
        full_seq = full_seq[:seq_len + 1]  # truncate to exact length

        # Split into input and target
        input_seq = full_seq[:-1]
        target = full_seq[-1]

        input_seqs.append(input_seq)
        targets.append(target)

    input_tensor = torch.tensor(input_seqs, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)

    return input_tensor, target_tensor

