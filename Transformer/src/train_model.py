""" 
python train_model.py [1] [2]
[1] (config_file)
[2] (strategie) : Identifiant de la stratégie d'embeddings à tester (1, 2 ou 3).
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd

from gpt import MyGPT 
from hp import HP_GPT
import utils


utils.set_seed(42)

# Détection du GPU ou CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} <fichier_config_json> <strategie>")
    print(f"Exemple: python {sys.argv[0]} config_gpt.json 1")
    sys.exit(1)

config_file = sys.argv[1]
strategy = int(sys.argv[2])

# 1. Chargement de la configuration et instanciation du modèle GPT
hp = HP_GPT(config_file)
model = MyGPT(hp)
model.to(device)

print(f"--- Lancement de l'entraînement GPT (Stratégie {strategy}) sur {device} ---")

# 2. Préparation des datasets avec ta fonction utils.prepare_gpt_dataset
train_examples = utils.prepare_gpt_dataset(
    filename=hp.train_filename,
    token_to_id=model.vocab.dico_voca,
    max_seq_length=model.hp.max_seq_length,
    pad_token="<pad>",
    unk_token="<unk>"
)

val_examples = utils.prepare_gpt_dataset(
    filename=hp.val_filename,
    token_to_id=model.vocab.dico_voca,
    max_seq_length=model.hp.max_seq_length,
    pad_token="<pad>",
    unk_token="<unk>"
)

# 3. Création des DataLoaders PyTorch
train_loader = DataLoader(
    train_examples,
    batch_size=model.hp.minibatch,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_examples,
    batch_size=model.hp.minibatch,
    shuffle=False,
    drop_last=False
)

# 4. Entraînement du modèle via la méthode fit()
exe_time, best_metrics = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=model.hp.nb_iter,
    strategy=strategy,
    lr=1e-3,
    patience=4
)

# 5. Sauvegarde des résultats et métriques d'hyperparamètres dans un fichier CSV
list_results_csv = []
list_results_csv.append({
    "num_heads": hp.num_heads,
    "num_layers": hp.num_layers,
    "emb_dim": hp.emb_dim,
    "max_seq_length": hp.max_seq_length,
    "minibatch": hp.minibatch,
    "exe_time_sec": exe_time,
    "best_epoch": best_metrics["epoch"],
    "best_train_loss": best_metrics["train_loss"],
    "best_val_loss": best_metrics["val_loss"],
    "best_gap": best_metrics["gap"],
})

output_csv_path = f"strategy_{strategy}_gpt.csv"

# Si le CSV existe déjà, on ajoute la ligne, sinon on le crée
if os.path.exists(output_csv_path):
    df_old = pd.read_csv(output_csv_path)
    df_new = pd.DataFrame(list_results_csv)
    df = pd.concat([df_old, df_new], ignore_index=True)
else:
    df = pd.DataFrame(list_results_csv)

df.to_csv(output_csv_path, index=False)
print(f"-> Métriques et résultats enregistrés dans : {output_csv_path}")