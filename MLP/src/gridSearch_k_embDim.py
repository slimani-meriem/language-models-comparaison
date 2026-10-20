"""
    Ce script réalise une recherche par grille (Grid Search) sur la taille du contexte
    ($k$) et de la dimension des embeddings. Pour chaque combinaison il adapte 
    dynamiquement la configuration, prépare le corpus, entraîne le modèle avec 
    accélération matérielle (PyTorch DataLoader multi-processus / GPU) et 
    enregistre au fur et à mesure les métriques de convergence dans des fichiers 
    JSON et CSV.

    python gridSsearch_k_embDim.py [1] [2]

    [1] (config_file) : Chemin vers le fichier de configuration JSON.
    [2] (strategie) : Identifiant de la stratégie d'embeddings à tester (1, 2 ou 3).
"""


import os
import argparse
import json
import itertools  # Importé pour générer le produit cartésien de la grille
import pandas as pd
import torch
import utils
from mlp import MLP, Hp

from torch.utils.data import DataLoader

# import de utils_data qui se trouve dans un autre dossier 
import sys
sys.path.append("../../data/tools/data_tools")
import utils_data


def preprare_launch(results_dir_name):
    # Pour la reproductibilité
    utils.set_seed(42)

    # Détection automatique du GPU (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositif de calcul détecté : {device}")

    os.makedirs(results_dir_name, exist_ok=True)
    return device


def apply_strategy(hp, k, emb_dim, strategy):
    """Modifie l'objet Hp en mémoire au lieu d'écrire sur le disque."""
    hp.k = k
    hp.freeze_embeddings = False
        
    if strategy == 1:
        hp.emb_dim = emb_dim
        hp.emb_filename = None
    if strategy == 2:
        hp.emb_dim = emb_dim
        hp.emb_filename = f"../../data/bpe_embeddings/bpe_embeddings_12k_{emb_dim}.txt"
    if strategy == 3: 
        hp.freeze_embeddings = True
        hp.emb_dim = emb_dim
        hp.emb_filename = f"../../data/bpe_embeddings/bpe_embeddings_12k_{emb_dim}.txt"


def launch_tests(CONFIG_FILE, strategy, device, results_dir_name):
    CORPUS = "../../data/corpus"
    TEMP_CORPUS = "../../data/corpus/temp"
    os.makedirs(CORPUS, exist_ok=True)

    dict_results_json = {}
    list_results_csv = []

    # Plage de valeurs pour la grille
    k_values = [3, 5 ]#8, 12, 15, 20, 30]
    emb_dim_values = [64]

    # Génère toutes les combinaisons possibles (Grid Search)
    grid_combinations = list(itertools.product(k_values, emb_dim_values))
    total_iterations = len(grid_combinations)

    print(f"Début du Grid Search : {total_iterations} combinaisons à tester.\n")

    for idx, (k, emb_dim) in enumerate(grid_combinations, start=1):

        print(f"\n Iteration {idx}/{total_iterations} | k = {k} | emb_dim = {emb_dim}", flush=True)


        utils_data.add_bos_eos_on_copy(
            folder_path=CORPUS,
            output_folder=TEMP_CORPUS,
            nb_bos=k,
            nb_eos=k,
            extension='_12k.bpe'
        )

        # Instanciation du modèle 
        hp = Hp(CONFIG_FILE)

        # Modification dynamique du contenu du fichier de configuration
        apply_strategy(hp, k, emb_dim, strategy)
        
        model = MLP(hp)

        # Envoi du modèle sur le gpu
        model = model.to(device)

        train_dataset, val_dataset = utils.prepare_data_no_overlap(model)

        # Créer les dataLoader pour accélérer l'execution des tests
        train_loader = DataLoader(
            train_dataset, 
            batch_size=model.hp.minibatch, 
            shuffle=True,          # mélange automatiquement à CHAQUE début d'époque
            num_workers=8,         # utilise les 8 CPUs du script run.sh
            pin_memory=True,       # pour accelerer la copie RAM -> GPU
            drop_last=True,         # drop le dernier batch si incomplet
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=model.hp.minibatch, 
            shuffle=False,         # pas besoin de mélanger la validation
            num_workers=8, 
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        exe_time, best_metrics = model.fit(
            train_loader, 
            val_loader, 
            epochs=model.hp.nb_iter, 
            strategy=strategy,
            lr=1e-3,
            alpha=0.5,
            patience = 30
        )

        # Remplissage des resultats 
        dict_results_json[str(idx)] = {
            "hyperparameters": {
                "k": k,
                "emb_dim": emb_dim
            },
            "exe_time": exe_time,
            "best_epoch_stats": {
                "epoch": best_metrics["epoch"],
                "train_loss": best_metrics["train_loss"],
                "val_loss": best_metrics["val_loss"],
                "gap": best_metrics["gap"],
                "score_combine": best_metrics["score"]
            }
        }
    
        list_results_csv.append({
            "iteration": idx,
            "k": k,
            "emb_dim": emb_dim,
            "exe_time_sec": exe_time,
            "best_epoch": best_metrics["epoch"],
            "best_train_loss": best_metrics["train_loss"],
            "best_val_loss": best_metrics["val_loss"],
            "best_gap": best_metrics["gap"],
            "best_score_combine": best_metrics["score"]
        })

        # Sauvegarde progressive à chaque itération pour ne rien perdre
        with open(f"{results_dir_name}/grid_search_k_embDim.json", "w", encoding="utf-8") as f:
            json.dump(dict_results_json, f, indent=4)

        df = pd.DataFrame(list_results_csv)
        df.to_csv(f"{results_dir_name}/grid_search_k_embDim.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparamétrage de la taille du contexte et la dimension de l'embedding par GridSearch.")

    parser.add_argument("config_file", type=str, help="Chemin vers le fichier de configuration .json (Attention! Spécifier le bon selon la stratégie à tester)")
    parser.add_argument("strategie", type=int, default=2, help="Stratégie d'embeddings à lancer (1,2 ou 3, par défaut à 2)")

    args = parser.parse_args()

    json_config_file = args.config_file
    strategy = args.strategie

    results_dir_name = f"resultats_grid_k_embDim_strategy_{strategy}"
    device = preprare_launch(results_dir_name)

    # Lancement du Grid Search
    launch_tests(json_config_file, strategy, device, results_dir_name)