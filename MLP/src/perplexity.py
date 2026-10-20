"""
    python perplexity.py \
    --model_path [1] \
    --config_path [2] \
    --test_file [3]
"""

import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from mlp import MLP


def evaluate_perplexity(model, test_loader, device):
    """
    Calcule la Cross-Entropy globale accumulée puis la perplexité.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # k premiers tokens = contexte, dernier token = cible
            prefix = batch[:, :model.hp.k]
            target = batch[:, -1]

            logits = model(prefix)
            
            loss = criterion(logits, target)
            total_loss += loss.item()
            total_tokens += target.size(0)

    mean_ce = total_loss / total_tokens
    ppl = math.exp(mean_ce)
    
    return mean_ce, ppl


def main():
    parser = argparse.ArgumentParser(description="Calcul de la perplexité MLP (ligne par ligne)")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le fichier .pt")
    parser.add_argument("--config_path", type=str, required=True, help="Chemin vers la config .json")
    parser.add_argument("--test_file", type=str, default=None, help="Fichier de test spécifique (optionnel)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # 1. Chargement du modèle
    print(f"Chargement du modèle depuis {args.model_path}...")
    model = MLP.load(args.model_path, args.config_path)
    model = model.to(device)

    # 2. Choix du fichier à évaluer (val par défaut ou fichier test)
    test_file = args.test_file if args.test_file else model.hp.val_filename

    # 3. Préparation des données isolées par ligne via utils
    print(f"Préparation des données via utils.prepare_data_no_overlap sur {test_file}...")
    # 3. Préparation des données pour la perplexité via utils
    test_dataset = utils.prepare_data_no_overlap_perplexity(model, test_file)

    test_loader = DataLoader(
        test_dataset,
        batch_size=model.hp.minibatch,
        shuffle=False
    )

    # 4. Calcul de la perplexité
    mean_loss, ppl = evaluate_perplexity(model, test_loader, device)

    print("\n" + "="*45)
    print(f"Nombre total d'exemples : {len(test_dataset)}")
    print(f"Cross-Entropy Moyenne   : {mean_loss:.4f} nats")
    print(f"PERPLEXITÉ (PPL)        : {ppl:.2f}")
    print("="*45 + "\n")


if __name__ == "__main__":
    main()