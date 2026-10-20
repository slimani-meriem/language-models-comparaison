"""
python evaluate.py \
    --mask_file [1] \
    --candidates_file [2] \
    --model_file [3] \
    --config_file [4] \
    --top_k [5]
"""

import sys
import os
import random
import torch
import torch.nn.functional as F
from gru import GRU  



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
    Génère une phrase mot par mot en conservant l'état caché (hidden state) du GRU.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Token initial <s>
    start_token_id = model.vocab.get_word_index2("<s>")
    
    # Entrée courante : [batch_size=1, sequence_length=1]
    current_input = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    h_t = None  # État caché initialisé à None

    tokens_generes = []

    with torch.no_grad():
        for _ in range(max_len):
            # Forward autoregressif sur un seul pas de temps
            logits, h_t = model.forward_evaluate(current_input, h_0=h_t)
            
            # Extraction des logits du dernier token et calcul des probabilités
            # logits a la forme (1, 1, vocab_size) -> logits[0, -1, :] donne (vocab_size,)
            prob_distrib = F.softmax(logits[0, -1, :], dim=0)

            # Tirage par la méthode de rejet
            argmax = random_rejet(prob_distrib)
            mot = model.vocab.get_word(argmax)

            # Fin de phrase si on rencontre </s>
            if mot == '</s>':
                break

            if mot != "<unk>":
                print(mot, end=" ", flush=True)
                tokens_generes.append(mot)

            # Prochaine entrée : uniquement l'indice du mot que l'on vient de générer
            current_input = torch.tensor([[argmax]], dtype=torch.long, device=device)

    print()  # Fin de ligne


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <fichier_modele_pt> <fichier_config_json> <nb_phrases>")
        print(f"Exemple: python {sys.argv[0]} best_models_strategy_1/GRU_h32_embDim200.pt config_gru.json 5")
        sys.exit(1)

    model_path = sys.argv[1]
    json_path = sys.argv[2]
    nb_phrases = int(sys.argv[3])

    if not os.path.exists(model_path) or not os.path.exists(json_path):
        print("Erreur : Le fichier de poids (.pt) ou le fichier JSON de config est introuvable.")
        sys.exit(1)

    print("Chargement du modèle GRU...")
    model = GRU.load(model_path, json_path)
    model.eval()

    print(f"\n--- Génération de {nb_phrases} phrase(s) avec le GRU ---\n")
    for i in range(nb_phrases):
        print(f"[{i+1}] ", end="")
        genere_phrase(model)