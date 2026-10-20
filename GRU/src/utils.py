import torch
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np

"""
    - Le corpus contient des phrases de longueurs différentes.
    - Les phrases sont deja balisées lors du traitement,
    - Nous prenons le corpus complet Nous formerons un tenseur où chaque ligne est une phrase,
    - Nous regrouperons dans des batchs les phrases dont la différence de longueur est de 5,
    - Nous ajouterons des paddings à la fin des phrases courtes avec <pad> (et pas </s>),
    - Nous ferons le split selons les tailles de batchs,
"""

def pad_sequences(
        filename,
        token_to_id,
        batch_size,
        max_diff=5,
        pad_token="<pad>",
        unk_token="<unk>",
        shuffle_batches=True
):
    
    pad_id = token_to_id[pad_token]
    unk_id = token_to_id[unk_token]

    sentences = []

    # Lire le fichier et transformer chaque phrase en un tenseur 
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 0 : continue

            ids = [token_to_id.get(tok, unk_id) for tok in tokens]

            sentence_tensor = torch.tensor(ids)
            sentences.append(sentence_tensor)

    # Trier les phrases par longueur 
    sentences = sorted(sentences, key=lambda x: len(x))

    # Creer les groupes de phrases dont la différence de longueur <= max_diff
    raw_batches = []
    current_batch = []
    current_min_len = None

    for sent in sentences:
        sent_len = len(sent)

        if len(current_batch) == 0:
            current_batch = [sent]
            current_min_len = sent_len

        elif len(current_batch) < batch_size and sent_len - current_min_len <= max_diff:
            current_batch.append(sent)

        else:
            raw_batches.append(current_batch)
            current_batch = [sent]
            current_min_len = sent_len

    if len(current_batch) > 0:
        raw_batches.append(current_batch)

    if shuffle_batches:
        random.shuffle(raw_batches)

    # Ajouter pading pour chaque batch
    padded_batches = []
    for batch in raw_batches :
        lengths = torch.tensor([len(sent) for sent in batch], dtype=torch.long)

        batch_padded = pad_sequence(
            batch,
            batch_first=True,
            padding_value=pad_id
        )

        padded_batches.append((batch_padded, lengths))

    return padded_batches


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # si plusieurs gpu sont utilisés
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False