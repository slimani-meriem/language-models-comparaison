import time
import torch
import torch.nn as nn
import torch.optim
import os
import json
import sys
from Vocab import Vocab
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy



# Fonction de perte
loss_func = torch.nn.CrossEntropyLoss()

class Hp:
    def __init__(self, json_file=None):
        # Valeurs par défaut
        self.k = 1
        self.dh = 200
        self.minibatch = 32
        self.nb_iter = 10
        self.freeze_embeddings = True
        self.emb_filename = None
        
        self.vocab_filename = None
        self.train_filename = None
        self.val_filename = None
        self.emb_dim = None

        if json_file:
            self.load_and_validate(json_file)

    def load_and_validate(self, json_file):
        if not os.path.exists(json_file):
            print(f" ERREUR : Le fichier de configuration '{json_file}' est introuvable.")
            sys.exit(1)

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f" ERREUR : Le fichier JSON est mal formé (erreur de syntaxe) :\n{e}")
            sys.exit(1)

        # AJUSTEMENT : On remplace "corpus_filename" par "train_filename"
        required_keys = ["k", "dh", "minibatch", "nb_iter", "train_filename", "val_filename", "vocab_filename"]
        missing = [key for key in required_keys if key not in data]

        if missing:
            print(f" ERREUR : Clé(s) manquante(s) dans le JSON : {', '.join(missing)}")
            print("Aide : Vérifiez que votre fichier contient bien ces paramètres.")
            sys.exit(1)

        # Embeddings
        if "emb_filename" not in data and "emb_dim" not in data:
            print(" ERREUR : Vous devez spécifier soit 'emb_filename' (embedding pré-entraînés) soit 'emb_dim' (taille des embeddings).")
            sys.exit(1)

        # Affectation dynamique des valeurs
        for key, value in data.items():
            setattr(self, key, value)

        print("Configuration chargée avec succès.")


class MLP(nn.Module):
    def __init__(self, hp):
        super(MLP, self).__init__()
        self.hp = hp

        # initialisation du vocab 
        if hp.emb_filename and hp.emb_filename != "null":
            # Cas avec des embeddings pré-entraînés
            self.vocab = Vocab(emb_filename=hp.emb_filename)
        else:
            # construction du vocab à partir d'un fichier .json issue de la tokenisation
            # la classe vocab sera traiter le fichier .json grâce à la méthode _convert_json_to_txt
            self.vocab = Vocab(vocab_filename=hp.vocab_filename, emb_dim=hp.emb_dim)

        # configuration de la couche d'embedding du modèle PyTorch
        if hp.emb_filename and hp.emb_filename != "null":
            self.E = torch.nn.Embedding.from_pretrained(self.vocab.matrice, freeze=hp.freeze_embeddings)
            current_emb_dim = self.vocab.matrice.shape[1]
            self.hp.emb_dim = current_emb_dim
        else:
            self.E = torch.nn.Embedding(self.vocab.vocab_size, hp.emb_dim)
            current_emb_dim = hp.emb_dim

        self.dx = hp.k * current_emb_dim

        # poids du modèle
        self.W1 = torch.nn.Parameter(torch.rand(self.dx, hp.dh))
        self.b1 = torch.nn.Parameter(torch.rand(hp.dh))

        self.W2 = torch.nn.Parameter(torch.rand(hp.dh, self.vocab.vocab_size))
        self.b2 = torch.nn.Parameter(torch.rand(self.vocab.vocab_size))
    """
    @staticmethod
    def load(model_path, json_path):
        config = Hp(json_path)
        model = MLP(config)
        weights = torch.load(model_path, weights_only=True, map_location='cpu')
        model.load_state_dict(weights)
        return model"""

    @staticmethod
    def load(model_path, json_path):
        config = Hp(json_path)
        weights = torch.load(model_path, weights_only=True, map_location='cpu')
        saved_vocab_size = weights["W2"].shape[1]
        model = MLP(config)
        if model.vocab.vocab_size != saved_vocab_size:
            print(f"! Ajustement du vocabulaire : Modèle = {model.vocab.vocab_size}, Checkpoint = {saved_vocab_size}")
            model.vocab.vocab_size = saved_vocab_size
            model.hp.vocab_size = saved_vocab_size
            
            # Redimensionner la couche d'embedding E
            current_emb_dim = weights["E.weight"].shape[1]
            model.E = torch.nn.Embedding(saved_vocab_size, current_emb_dim)
            
            # Redimensionner W2 et b2
            dh = config.dh
            model.W2 = torch.nn.Parameter(torch.rand(dh, saved_vocab_size))
            model.b2 = torch.nn.Parameter(torch.rand(saved_vocab_size))

        model.load_state_dict(weights)
        return model


    def display_hp(self):
        print("k =          ", self.hp.k)
        print("dh =         ", self.hp.dh)
        print("emb_dim =    ", self.hp.emb_dim)
        print("vocab_size = ", self.hp.vocab_size)
        print("dx =         ", self.dx)
        print("nb_iter =    ", self.hp.nb_iter)
        print("minibatch =  ", self.hp.minibatch)
        print("freeze_embeddings =  ", self.hp.freeze_embeddings)
        print("fichier d'apprentissage =  ", self.hp.corpus_filename)
        print("fichier d'embeddings =  ", self.hp.emb_filename)


    def forward(self, x):
        # Transformer les données
        x_emb = self.E(x).view(x.size(0), self.hp.k * self.hp.emb_dim)
        hprime =  torch.matmul(x_emb, self.W1) + self.b1
        h = torch.nn.functional.relu(hprime)
        yprime =  torch.matmul(h, self.W2) + self.b2
        return yprime


    def fit(self, train_loader, val_loader, epochs, strategy, lr=1e-3, patience=12):
        start_time = time.time()  # --- Début de l'exécution ---

        optim = torch.optim.Adam(self.parameters(), lr=lr)
            
        best_val_loss = np.inf
        best_model_weights = None
            
        patience_counter = 0
            
        # Détection du device sur lequel se trouve le modèle (GPU ou CPU)
        device = next(self.parameters()).device
            
        # Structure pour conserver les meilleures métriques correspondantes
        best_metrics = {
            "train_loss": 0.0, 
            "val_loss": 0.0, 
            "gap": 0.0, 
            "epoch": 0
        }

        for epoch in range(epochs):
            # --- PHASE D'ENTRAÎNEMENT ---
            self.train()
            running_train_loss = 0.0

            for batch in train_loader:
                # --- ENVOI DU BATCH SUR LE GPU ---
                batch = batch.to(device, non_blocking=True) 

                prefix = batch[:, :self.hp.k]  # Les k premiers indices = contexte
                target = batch[:, -1]          # Le dernier indice = cible

                optim.zero_grad()
                yprime = self.forward(prefix)
                loss = loss_func(yprime, target)
                running_train_loss += loss.item()

                loss.backward()
                optim.step()

            # --- PHASE DE VALIDATION ---
            self.eval()
            running_val_loss = 0.0

            with torch.no_grad():  
                for batch in val_loader:
                    batch = batch.to(device)

                    prefix = batch[:, :self.hp.k]
                    target = batch[:, -1]
                        
                    val_pred = self.forward(prefix)
                    batch_loss = loss_func(val_pred, target)
                    running_val_loss += batch_loss.item()

            # Calcul des moyennes par batch sur toute l'époque
            epoch_train_loss = running_train_loss / len(train_loader)
            epoch_val_loss = running_val_loss / len(val_loader)
            gap = abs(epoch_train_loss - epoch_val_loss)

            print(f"Époque {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Gap: {gap:.4f}", flush=True)

            # --- GESTION DE LA PATIENCE SUR LA VAL_LOSS ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                    
                # Deep copy des poids
                best_model_weights = copy.deepcopy(self.state_dict())
                    
                best_metrics["train_loss"] = epoch_train_loss
                best_metrics["val_loss"] = epoch_val_loss
                best_metrics["gap"] = gap
                best_metrics["epoch"] = epoch + 1
                print(f" --> Nouvelle meilleure Val Loss détectée ({epoch_val_loss:.4f})", flush=True)
            else:
                patience_counter += 1
                print(f" --> Stagnation détectée. Patience : {patience_counter}/{patience}", flush=True)
        
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Pas d'amélioration de la val_loss depuis {patience} époques. Arrêt à l'époque {epoch+1}.", flush=True)
                break

        # --- FIN DES ÉPOQUES / ARRÊT : SAUVEGARDE DU MEILLEUR MODÈLE ---
        os.makedirs(f"best_models_strategy_{strategy}", exist_ok=True)
        k = self.hp.k
        emb_dim = self.hp.emb_dim

        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)
            torch.save(best_model_weights, f"best_models_strategy_{strategy}/best_model_k{k}_emDim{emb_dim}.pt")
            print("-> Fichier .pt du Best Model sauvegardé avec succès.", flush=True)

        exe_time = time.time() - start_time

        return exe_time, best_metrics