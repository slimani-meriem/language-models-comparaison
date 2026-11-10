""" 
python set_num_heads.py [1] [2]
[1] (config_file)
[2] (strategie) : Identifiant de la stratégie d'embeddings à tester (1, 2 ou 3).
"""

import utils
import torch
import os
from hp import HP_GPT
from gpt import MyGPT
from torch.utils.data import DataLoader
import pandas as pd
import argparse

EMB_DIM = 528 
NUM_LAYERS = 1


def prepare_launch(results_dir_name):
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    os.makedirs(results_dir_name, exist_ok=True)
    return device

def apply_params(hp, num_layers, num_heads):
    hp.num_layers = num_layers
    hp.num_heads = num_heads

def launch_tests(CONFIG_FILE, strategy, device, results_dir_name, vocab_size=12000):
    results = []
    idx = 1

    heads_values = [1, 2, 4, 6, 8, 12]

    for num_heads in heads_values :

        hp = HP_GPT(CONFIG_FILE)

        hp.strategy = strategy
        hp.emb_dim = EMB_DIM
        hp.num_heads = num_heads

        if strategy == 1:
            hp.emb_filename = None

        elif strategy in [2,3]:
            hp.emb_filename = (
                f"../../data/bpe_embeddings/"
                f"bpe_embeddings_{vocab_size//1000}k_{EMB_DIM}.txt"
            )

            if strategy == 3:
                hp.freeze_embeddings = True
            else:
                hp.freeze_embeddings = False


        model = MyGPT(hp)
        model = model.to(device)

        train_examples = utils.prepare_gpt_dataset(
            hp.train_filename,
            model.vocab.dico_voca,
            hp.max_seq_length
        )

        val_examples = utils.prepare_gpt_dataset(
            hp.val_filename,
            model.vocab.dico_voca,
            hp.max_seq_length
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


        exe_time, best_metrics = model.fit(
            train_loader,
            val_loader,
            epochs=hp.nb_iter,
            strategy=strategy,
            lr=1e-4,
            patience=10
        )


        results.append(
            {
                "iteration": idx,
                "embedding_dim": EMB_DIM,
                "num_layers": NUM_LAYERS,
                "num_heads": num_heads,
                "exe_time_sec": float(exe_time),
                "best_epoch": int(best_metrics["epoch"]),
                "best_train_loss": float(best_metrics["train_loss"]),
                "best_val_loss": float(best_metrics["val_loss"]),
                "best_gap": float(best_metrics["gap"])
            }
        )


        csv_path = os.path.join(
            results_dir_name,
            f"search_heads_strategy_{strategy}.csv"
        )

        pd.DataFrame(results).to_csv(
            csv_path,
            index=False
        )

        print(f"Résultats sauvegardés : {csv_path}")

        idx += 1

        del model
        del train_loader
        del val_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Grid search num_layers et num_heads pour GPT"
    )

    parser.add_argument(
        "config_file",
        type=str
    )

    parser.add_argument(
        "strategie",
        type=int,
        choices=[1,2,3]
    )

    args = parser.parse_args()

    strategy = args.strategie

    results_dir_name = (
        f"resultats_layers_heads_strategy_{strategy}"
    )

    device = prepare_launch(results_dir_name)

    launch_tests(
        args.config_file,
        strategy,
        device,
        results_dir_name
    )