""" 
    python perplexity.py \
    --model_path [1] \
    --config_path [2] \
    --test_file [3] \
    --batch_size [4]
"""

import math
import argparse
import torch
import torch.nn as nn

from gru import GRU
from utils import pad_sequences


def evaluate_perplexity(model, test_loader, device):
    """
    Calcule la Cross-Entropy globale accumulée puis la perplexité pour le GRU.
    Ignores les tokens de padding dans la perte.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id, reduction='sum')
    
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch, lengths in test_loader:
            batch = batch.to(device)
            lengths = lengths.to(device)

            # Entrées = du 1er au navant-dernier token
            # Cibles = du 2ème au dernier token
            prefix = batch[:, :-1]
            target = batch[:, 1:]
            prefix_lengths = lengths - 1

            # Passage avant dans le GRU
            logits, _ = model.forward(prefix, prefix_lengths)
            
            # Reprojection pour le calcul du Cross-Entropy
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )

            total_loss += loss.item()
            # Compte uniquement les tokens valides (hors padding)
            total_tokens += (target != model.pad_id).sum().item()

    if total_tokens == 0:
        raise ValueError("Aucun token valide trouvé pour l'évaluation.")

    mean_ce = total_loss / total_tokens
    ppl = math.exp(mean_ce)
    
    return mean_ce, ppl


def main():
    parser = argparse.ArgumentParser(description="Calcul de la perplexité GRU")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le fichier .pt")
    parser.add_argument("--config_path", type=str, required=True, help="Chemin vers la config .json")
    parser.add_argument("--test_file", type=str, required=True, help="Chemin vers le fichier texte de test")
    parser.add_argument("--batch_size", type=int, default=256, help="Taille des mini-lots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # 1. Chargement du modèle GRU
    print(f"Chargement du modèle depuis {args.model_path}...")
    model = GRU.load(args.model_path, args.config_path)
    model = model.to(device)

    # 2. Récupération du vocabulaire
    token_to_id = model.vocab.dico_voca

    # 3. Préparation des données récurrentes (rembourrage par séquences)
    print(f"Préparation des données via pad_sequences sur {args.test_file}...")
    test_loader = pad_sequences(
        filename=args.test_file,
        token_to_id=token_to_id,
        batch_size=args.batch_size,
        max_diff=5,
        pad_token="<pad>",
        unk_token="<unk>",
        shuffle_batches=False
    )

    # 4. Calcul de la perplexité
    mean_loss, ppl = evaluate_perplexity(model, test_loader, device)

    print("\n" + "="*45)
    print(f"Cross-Entropy Moyenne : {mean_loss:.4f} nats")
    print(f"PERPLEXITÉ (PPL)      : {ppl:.2f}")
    print("="*45 + "\n")


if __name__ == "__main__":
    main()