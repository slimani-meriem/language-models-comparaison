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
from torch.utils.data import DataLoader

from gpt import MyGPT
from utils import prepare_gpt_dataset


def evaluate_perplexity(model, test_loader, device):
    """
    Calcule la Cross-Entropy globale accumulée puis la perplexité pour MyGPT.
    Ignores les tokens de padding dans la perte.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id, reduction='sum')
    
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for prefix, target in test_loader:
            prefix = prefix.to(device)
            target = target.to(device)

            # Forward pass -> (B, V, T)
            logits = model(prefix)
            
            # Alignement des dimensions : (B, V, T) -> (B, T, V) -> (-1, V)
            logits_flat = logits.transpose(1, 2).reshape(-1, logits.size(1))
            target_flat = target.reshape(-1)

            loss = criterion(logits_flat, target_flat)

            total_loss += loss.item()
            total_tokens += (target != model.pad_id).sum().item()

    if total_tokens == 0:
        raise ValueError("Aucun token valide trouvé pour l'évaluation.")

    mean_ce = total_loss / total_tokens
    ppl = math.exp(mean_ce)
    
    return mean_ce, ppl


def main():
    parser = argparse.ArgumentParser(description="Calcul de la perplexité Transformer (MyGPT)")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le fichier .pt")
    parser.add_argument("--config_path", type=str, required=True, help="Chemin vers la config .json")
    parser.add_argument("--test_file", type=str, required=True, help="Chemin vers le fichier texte de test")
    parser.add_argument("--batch_size", type=int, default=256, help="Taille des mini-lots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # 1. Chargement du modèle MyGPT
    print(f"Chargement du modèle depuis {args.model_path}...")
    model = MyGPT.load(args.model_path, args.config_path)
    model = model.to(device)

    # 2. Récupération du vocabulaire
    token_to_id = model.vocab.dico_voca

    # 3. Préparation des données via prepare_gpt_dataset
    print(f"Préparation des données via prepare_gpt_dataset sur {args.test_file}...")
    dataset = prepare_gpt_dataset(
        filename=args.test_file,
        token_to_id=token_to_id,
        max_seq_length=model.hp.max_seq_length,
        pad_token="<pad>",
        unk_token="<unk>"
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 4. Calcul de la perplexité
    mean_loss, ppl = evaluate_perplexity(model, test_loader, device)

    print("\n" + "="*45)
    print(f"Cross-Entropy Moyenne : {mean_loss:.4f} nats")
    print(f"PERPLEXITÉ (PPL)      : {ppl:.2f}")
    print("="*45 + "\n")


if __name__ == "__main__":
    main()