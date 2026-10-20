"""
    python train_model.py [1] [2]
    [1] (config_file) : Chemin vers le fichier de configuration JSON.
    [2] (strategy) : Identifiant de la stratégie d'embeddings à utiliser (1, 2 ou 3).
"""
from gru import GRU
from hp import HP_GRU
import utils
import torch 
import sys
from torch.utils.data import DataLoader
import pandas as pd


utils.set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_file = sys.argv[1]
strategy = sys.argv[2]

hp = HP_GRU(config_file)
model = GRU(hp)

model.to(device)

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
        )

val_loader = DataLoader(
            val_batches,
            batch_size=None,
            shuffle=False,
        )

exe_time, best_metrics = model.fit(
            train_loader,
            val_loader,
            epochs=model.hp.nb_iter,
            strategy=strategy,
            lr=1e-3,
            patience=4
)

list_results_csv = []
list_results_csv.append({
        "h": 256,
        "embedding_dim": 256,
        "num_layers": 1,
        "exe_time_sec": exe_time,
        "best_epoch": best_metrics["epoch"],
        "best_train_loss": best_metrics["train_loss"],
        "best_val_loss": best_metrics["val_loss"],
        "best_gap": best_metrics["gap"],
    })

df = pd.DataFrame(list_results_csv)
df.to_csv(f"strategy_{strategy}.csv", index=False)