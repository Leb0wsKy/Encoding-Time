import torch

def generate_delayed_match_data(delay=10, vocab_size=10, batch_size=32):
    first_token = torch.randint(1, vocab_size, (batch_size, 1))
    noise = torch.randint(1, vocab_size, (batch_size, delay))
    query_token = torch.zeros(batch_size, 1).long()  # "0" marks the query

    input_seq = torch.cat([first_token, noise, query_token], dim=1)
    target_seq = torch.zeros_like(input_seq)
    target_seq[:, -1] = first_token.squeeze()

    return input_seq, target_seq
