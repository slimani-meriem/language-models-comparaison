""" 
python set_emb_dim.py [1] [2]
[1] (config_file)
[2] (strategie) : Identifiant de la stratégie d'embeddings à tester (1, 2 ou 3).
"""


import utils
import torch
import os
import argparse
import pandas as pd

from torch.utils.data import DataLoader

from hp import HP_GPT
from gpt import MyGPT


NUM_LAYERS = 1
NUM_HEADS = 1


def prepare_launch(results_dir_name):
    # Reproductibilité
    utils.set_seed(42)

    # GPU si disponible
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositif détecté : {device}")

    os.makedirs(results_dir_name, exist_ok=True)

    return device


def apply_strategy(hp, emb_dim, strategy):
    """
    Modifie les hyperparamètres en mémoire.
    """

    hp.emb_dim = emb_dim
    hp.strategy = strategy

    hp.num_layers = NUM_LAYERS
    hp.num_heads = NUM_HEADS

    hp.freeze_embeddings = False

    if strategy == 1:
        hp.emb_filename = None

    elif strategy == 2:
        hp.freeze_embeddings = False
        hp.emb_filename = (
            f"../../data/bpe_embeddings/"
            f"bpe_embeddings_12k_{emb_dim}.txt"
        )

    elif strategy == 3:
        hp.freeze_embeddings = True
        hp.emb_filename = (
            f"../../data/bpe_embeddings/"
            f"bpe_embeddings_12k_{emb_dim}.txt"
        )


def launch_tests(
        CONFIG_FILE,
        strategy,
        device,
        results_dir_name
):

    results = []
    emb_dim_values = [48, 72, 144, 264, 528, 768]
    iteration = 1

    for emb_dim in emb_dim_values:
        print("=" * 70)
        print(f"Embedding dimension = {emb_dim}")
        print("=" * 70)

        hp = HP_GPT(CONFIG_FILE)

        apply_strategy(
            hp,
            emb_dim,
            strategy
        )

        model = MyGPT(hp).to(device)

        # ------------- Préparation des données ------------------
        train_examples = utils.prepare_gpt_dataset(
            filename=hp.train_filename,
            token_to_id=model.vocab.dico_voca,
            max_seq_length=hp.max_seq_length
        )

        val_examples = utils.prepare_gpt_dataset(
            filename=hp.val_filename,
            token_to_id=model.vocab.dico_voca,
            max_seq_length=hp.max_seq_length
        )

        train_loader = DataLoader(
            train_examples,
            batch_size=hp.minibatch,
            shuffle=True
        )

        val_loader = DataLoader(
            val_examples,
            batch_size=hp.minibatch,
            shuffle=False
        )

        # ---------------- Entraînement -----------
        exe_time, best_metrics = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=hp.nb_iter,
            strategy=strategy,
            lr=1e-4,
            patience=30
        )

        # ---------- Sauvegarde résultats ---------------
        results.append({
            "iteration": iteration,
            "embedding_dim": emb_dim,
            "num_layers": hp.num_layers,
            "num_heads": hp.num_heads,
            "exe_time_sec": exe_time,
            "best_epoch": best_metrics["epoch"],
            "best_train_loss": best_metrics["train_loss"],
            "best_val_loss": best_metrics["val_loss"],
            "best_gap": best_metrics["gap"]
        })

        csv_path = os.path.join(
            results_dir_name,
            f"search_embedding_strategy_{strategy}.csv"
        )

        pd.DataFrame(results).to_csv(
            csv_path,
            index=False
        )

        print(
            f"Résultats enregistrés dans {csv_path}"
        )

        iteration += 1

        del model
        del train_loader
        del val_loader
        del train_examples
        del val_examples

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Recherche de la meilleure dimension d'embedding pour GPT."
    )

    parser.add_argument(
        "config_file",
        type=str
    )

    parser.add_argument(
        "strategie",
        type=int,
        choices=[1, 2, 3]
    )

    args = parser.parse_args()

    results_dir = (
        f"resultats_serach_embedding_strategy_{args.strategie}"
    )

    device = prepare_launch(results_dir)

    launch_tests(
        CONFIG_FILE=args.config_file,
        strategy=args.strategie,
        device=device,
        results_dir_name=results_dir
    )