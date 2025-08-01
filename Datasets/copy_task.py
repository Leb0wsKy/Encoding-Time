import torch
import random

def generate_copy_data(seq_len=100, vocab_size=10, batch_size=32):
    X = torch.randint(1, vocab_size, (batch_size, seq_len))
    delimiter = torch.zeros(batch_size, 1).long()  
    pad = torch.zeros(batch_size, seq_len).long()
    
    input_seq = torch.cat([X, delimiter, pad], dim=1)
    target_seq = torch.cat([pad, delimiter, X], dim=1)
    
    return input_seq, target_seq
