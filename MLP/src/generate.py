"""
    Ce script génère de façon autorégressive des phrases mot par mot à partir d'un 
    modèle MLP entraîné, en utilisant une méthode d'échantillonnage par rejet. 
    À chaque étape, la fenêtre glissante des $k$ derniers tokens sert d'entrée au 
    MLP pour calculer la distribution Softmax du mot suivant, puis la fenêtre de 
    contexte est mise à jour jusqu'à l'obtention du token de fin de séquence </s>.

    python generate.py [1] [2] [3]

    [1] : Chemin vers le fichier modèle (.pt).
    [2] : Chemin vers le fichier de configuration JSON contenant les hyperparamètres.   
    [3] : Nombre de phrases à générer.

"""


import sys
import os
import random
import torch
from mlp import MLP, Hp  

def random_rejet(distrib):
    """
    Échantillonnage par rejet sur une distribution de probabilités.
    distrib : Tenseur 1D PyTorch contenant les probabilités (après Softmax)
    """
    N = len(distrib)
    # torch.max() pour avoir la valeur max et non l'index (argmax)
    vmax = torch.max(distrib).item()
    
    while True:
        k = random.randint(0, N - 1)
        x = random.random() * vmax
        # Rejet/Acceptation
        if x <= distrib[k].item():
            return k


def genere_phrase(model):
    """
    Génère une phrase mot par mot en utilisant le MLP et l'échantillonnage par rejet.
    """
    model.eval()
    k_context = model.hp.k
    
    # 1. Initialisation de la séquence d'entrée avec 'k' tokens <s>
    start_token_id = model.vocab.get_word_index2("<s>")
    prefix = [start_token_id] * k_context

    tokens_generes = []

    while True:
        # 2. Préparation du tenseur pour le forward : forme (batch_size=1, k)
        input_tensor = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
        
        # Passer sur le même device que le modèle (CPU ou GPU)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            # Evaluation du modèle
            logits = model(input_tensor)[0]  # Extraction du 1er élément du batch (taille = vocab_size)
            prob_distrib = torch.nn.functional.softmax(logits, dim=0)

        # 3. Tirage du mot suivant par la méthode de rejet
        argmax = random_rejet(prob_distrib)
        mot = model.vocab.get_word(argmax)

        # 4. Traitement du mot généré
        if mot != "<unk>":
            if mot == '</s>':
                break
            
            print(mot, end=" ", flush=True)
            tokens_generes.append(mot)

            # 5. Mise à jour de la fenêtre de contexte (décalage à gauche)
            prefix = prefix[1:] + [argmax]

    print()  # Fin de phrase


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <fichier_modele_pt> <fichier_config_json> <nb_phrases>")
        print(f"Exemple: python {sys.argv[0]} best_models_strategy_1/best_model_k3_emDim200.pt config.json 5")
        sys.exit(1)

    model_path = sys.argv[1]
    json_path = sys.argv[2]
    nb_phrases = int(sys.argv[3])

    if not os.path.exists(model_path) or not os.path.exists(json_path):
        print("Erreur : Le fichier de poids (.pt) ou de configuration (.json) est introuvable.")
        sys.exit(1)

    print("Chargement du modèle MLP...")
    model = MLP.load(model_path, json_path)
    
    # Passage en mode évaluation
    model.eval()

    print(f"\n--- Génération de {nb_phrases} phrase(s) ---\n")
    for i in range(nb_phrases):
        print(f"[{i+1}] ", end="")
        genere_phrase(model)