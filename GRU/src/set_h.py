"""
    Hyperparametrage de la taille du vecteur d'état 
    
    python set_h.py [1] [2]
        
        [1] (config_file) : Chemin vers le fichier de configuration JSON.
        [2] (strategie) : Identifiant de la stratégie d'embeddings à tester (1, 2 ou 3).
"""


import utils
import torch 
import os
from hp import HP_GRU
from gru import GRU
from torch.utils.data import DataLoader
import argparse
import pandas as pd

EMB_DIM = 32 

def preprare_launch(results_dir_name):
    # Pour la reproductibilité
    utils.set_seed(42)

    # Détection automatique du GPU (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositif de calcul détecté : {device}")

    os.makedirs(results_dir_name, exist_ok=True)
    return device


def launch_tests(CONFIG_FILE, strategy, device, results_dir_name):
    #dict_results_json = {}
    list_results_csv = []
    idx = 1
    h_values = [1, 16] #, 64, 128, 256, 512, 1024, 2048]

    for h in h_values :
        hp = HP_GRU(CONFIG_FILE)
        hp.hidden_size = h

        model = GRU(hp)

        model = model.to(device)

        train_batches = utils.pad_sequences(
            hp.train_filename,
            model.vocab.dico_voca,
            batch_size=model.hp.minibatch,
            max_diff=5,
            pad_token="<pad>",
            unk_token="<unk>",
            shuffle_batches=True
        )

        val_batches = utils.pad_sequences(
            hp.val_filename,
            model.vocab.dico_voca,
            batch_size=model.hp.minibatch,
            max_diff=5,
            pad_token="<pad>",
            unk_token="<unk>",
            shuffle_batches=False
        )

        train_loader = DataLoader(
            train_batches,
            batch_size=None,
            shuffle=True,
            #collate_fn=lambda batch: batch[0]
        )

        val_loader = DataLoader(
            val_batches,
            batch_size=None,
            shuffle=False,
            #collate_fn=lambda batch: batch[0]
        )

        first_batch = next(iter(train_loader))

        if isinstance(first_batch, (tuple, list)):
            print(
                "Format d'un élément du DataLoader :",
                [
                    getattr(element, "shape", type(element))
                    for element in first_batch
                ]
            )
        else:
            print(
                "Format d'un élément du DataLoader :",
                first_batch.shape
            )


        exe_time, best_metrics = model.fit(
            train_loader,
            val_loader,
            epochs=model.hp.nb_iter,
            strategy=strategy,
            lr=1e-3,
            patience=30
        )
        """
        result_iteration = {
            "hyperparameters": {
                "h": h,
                "embedding_dim": model.E.embedding_dim,
                "num_layers": model.hp.num_layers,
            },
            "exe_time": float(exe_time),
            "best_epoch_stats": {
                "epoch": int(best_metrics["epoch"]),
                "train_loss": float(best_metrics["train_loss"]),
                "val_loss": float(best_metrics["val_loss"]),
                "gap": float(best_metrics["gap"]),
            }
        }"""

        list_results_csv.append({
            "iteration": idx,
            "h": h,
            "embedding_dim": model.E.embedding_dim,
            "num_layers": model.hp.num_layers,
            "exe_time_sec": float(exe_time),
            "best_epoch": int(best_metrics["epoch"]),
            "best_train_loss": float(best_metrics["train_loss"]),
            "best_val_loss": float(best_metrics["val_loss"]),
            "best_gap": float(best_metrics["gap"]),
        })

        idx += 1
        """json_path = os.path.join(
            results_dir_name,
            f"search_h_strategy_{strategy}.json"
        )"""

        csv_path = os.path.join(
            results_dir_name,
            f"search_h_strategy_{strategy}.csv"
        )

        """with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                json_path,
                f,
                indent=4,
                ensure_ascii=False
            )"""
        
        df = pd.DataFrame(list_results_csv)
        df.to_csv(csv_path, index=False)

        print(
            f"\nRésultats de l'itération {idx} enregistrés :"
            f"\n- CSV  : {csv_path}"
        )

        del model
        del train_loader
        del val_loader
        del train_batches
        del val_batches

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Recherche de la meilleure taille de couche cachée h "
            "pour le modèle GRU."
        )
    )

    parser.add_argument(
        "config_file",
        type=str,
        help=(
            "Chemin vers le fichier de configuration JSON, "
            "par exemple configs/config_2.json"
        )
    )

    parser.add_argument(
        "strategie",
        type=int,
        choices=[1, 2, 3],
        help="Stratégie d'embeddings à lancer : 1, 2 ou 3."
    )

    args = parser.parse_args()

    json_config_file = args.config_file
    strategy = args.strategie

    results_dir_name = (
        f"resultats_grid_h_strategy_{strategy}"
    )

    device = preprare_launch(results_dir_name)

    launch_tests(
        json_config_file,
        strategy,
        device,
        results_dir_name
    )

