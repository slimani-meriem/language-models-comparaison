"""
    Ce script entraîne le modèle MLP avec sa meilleure configuration 
    d'hyperparamètres sur une stratégie d'embeddings donnée. Il prépare les données, 
    effectue l'apprentissage via des DataLoader PyTorch optimisés 
    (GPU, multi-processus), puis sauvegarde les métriques finales 
    (losses, gap, temps) dans un fichier CSV.

    python train_model.py [1] [2]

    [1] (config_file) : Chemin vers le fichier de configuration JSON.
    [2] (strategy) : Identifiant de la stratégie d'embeddings à utiliser (1, 2 ou 3).
"""


from mlp import MLP, Hp
import utils
import torch 
import sys
from torch.utils.data import DataLoader
import pandas as pd


utils.set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_file = sys.argv[1]
strategy = sys.argv[2]

hp = Hp(config_file)
model = MLP(hp)

model.to(device)

train_dataset, val_dataset = utils.prepare_data(model)
        # Créer les dataLoader pour accélérer l'execution des tests
train_loader = DataLoader(
    train_dataset, 
    batch_size=model.hp.minibatch, 
    shuffle=True,          # mélange automatiquement à CHAQUE début d'époque
    num_workers=8,         # utilise les 8 CPUs du script run.sh
    pin_memory=True,       # pour accelerer la copie RAM -> GPU
    drop_last=True         # drop le dernier batch si incomplet
)

val_loader = DataLoader(
        val_dataset, 
        batch_size=model.hp.minibatch, 
        shuffle=False,         # pas besoin de mélanger la validation
        num_workers=8, 
        pin_memory=True,
        drop_last=True
)


exe_time, best_metrics = model.fit(
    train_loader, 
    val_loader, 
    model.hp.nb_iter, 
    strategy,
    lr=1e-3,
    patience = 4
)

list_results_csv = []
list_results_csv.append({
        "k": 3,
        "emb_dim": 32,
        "exe_time_sec": exe_time,
        "best_epoch": best_metrics["epoch"],
        "best_train_loss": best_metrics["train_loss"],
        "best_val_loss": best_metrics["val_loss"],
        "best_gap": best_metrics["gap"],
    })

df = pd.DataFrame(list_results_csv)
df.to_csv(f"strategy_{strategy}.csv", index=False)