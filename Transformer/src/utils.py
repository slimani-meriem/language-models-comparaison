import torch 
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import numpy as np

def prepare_gpt_dataset(
        filename,
        token_to_id,
        max_seq_length,
        pad_token="<pad>",
        unk_token="<unk>"
):
    pad_id = token_to_id[pad_token]
    unk_id = token_to_id[unk_token]
    examples = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue

            ids = [token_to_id.get(tok, unk_id) for tok in tokens]

            ###################################################
            # Cas 1 : phrase plus courte que max_seq_length
            ###################################################

            if len(ids) <= max_seq_length:
                x = ids[:-1]
                y = ids[1:]
                x = x + [pad_id] * (max_seq_length - len(x))
                y = y + [pad_id] * (max_seq_length - len(y))
                examples.append(
                    (
                        torch.tensor(x),
                        torch.tensor(y)
                    )
                )

            ###################################################
            # Cas 2 : phrase longue
            ###################################################

            else:
                for i in range(len(ids) - max_seq_length):
                    x = ids[i:i+max_seq_length]
                    y = ids[i+1:i+max_seq_length+1]
                    examples.append(
                        (
                            torch.tensor(x),
                            torch.tensor(y)
                        )
                    )

    return examples


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # si plusieurs gpu sont utilisés
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False