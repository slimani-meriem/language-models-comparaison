"""
python generate_gpt.py [1] [2] [3]

[1] (fichier_modele_pt) : Chemin vers le fichier modèle Transformer (.pt).
[2] (fichier_config_json) : Chemin vers le fichier de configuration JSON.
[3] (nb_phrases) : Nombre de phrases à générer.

"""

import sys
import os
import random
import torch
import torch.nn.functional as F
from gpt import MyGPT 


def random_rejet(distrib):
    """
    Échantillonnage par rejet sur une distribution 1D de probabilités (Softmax).
    """
    N = len(distrib)
    vmax = torch.max(distrib).item()
    
    while True:
        k = random.randint(0, N - 1)
        x = random.random() * vmax
        if x <= distrib[k].item():
            return k


def genere_phrase(model, max_len=100):
    """
    Génère une phrase mot par mot en utilisant le Transformer (MyGPT).
    """
    model.eval()
    device = next(model.parameters()).device
    max_seq_len = model.hp.max_seq_length  # Limite de l'encodage positionnel (128)

    # 1. Initialisation avec le token de début <s>
    start_token_id = model.vocab.get_word_index2("<s>")
    prefix_indices = [start_token_id]

    tokens_generes = []

    with torch.no_grad():
        for _ in range(max_len):
            # 2. Gestion de la fenêtre contextuelle (si la séquence dépasse max_seq_length)
            context = prefix_indices[-max_seq_len:]
            
            # Préparation du tenseur d'entrée : (batch_size=1, seq_length)
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)

            # 3. Forward du Transformer
            # En sortie, logits a la forme (1, vocab_size, seq_length) en raison du swapaxes dans forward()
            logits = model(input_tensor)

            # 4. Extraction des logits du tout dernier token généré
            last_token_logits = logits[0, :, -1]  # Forme : (vocab_size,)
            prob_distrib = F.softmax(last_token_logits, dim=0)

            # 5. Tirage par la méthode de rejet
            argmax = random_rejet(prob_distrib)
            mot = model.vocab.get_word(argmax)

            # Arrêt si fin de phrase </s>
            if mot == '</s>':
                break

            if mot != "<unk>":
                print(mot, end=" ", flush=True)
                tokens_generes.append(mot)

            # 6. Ajout du nouveau token à l'historique
            prefix_indices.append(argmax)

    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <fichier_modele_pt> <fichier_config_json> <nb_phrases>")
        print(f"Exemple: python {sys.argv[0]} best_models_strategy_1/GPT_embDim32.pt config_gpt.json 5")
        sys.exit(1)

    model_path = sys.argv[1]
    json_path = sys.argv[2]
    nb_phrases = int(sys.argv[3])

    if not os.path.exists(model_path) or not os.path.exists(json_path):
        print("Erreur : Le fichier de poids (.pt) ou le fichier JSON de config est introuvable.")
        sys.exit(1)

    print("Chargement du modèle Transformer (MyGPT)...")
    model = MyGPT.load(model_path, json_path)
    model.eval()

    print(f"\n--- Génération de {nb_phrases} phrase(s) avec le Transformer ---\n")
    for i in range(nb_phrases):
        print(f"[{i+1}] ", end="")
        genere_phrase(model)