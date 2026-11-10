import torch 
import torch.nn as nn
import transformer as T
from Vocab import Vocab
from hp import HP_GPT
import numpy as np
import time 
import utils
import copy 
import os

class MyGPT(nn.Module):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        if hp.strategy == 1 :
            self.vocab = Vocab(vocab_filename=hp.vocab_filename, emb_dim=hp.emb_dim)
            self.pad_id = self.vocab.pad_id
        
            self.E = torch.nn.Embedding(
                self.vocab.vocab_size, 
                hp.emb_dim, 
                padding_idx=self.pad_id
            )
        
            embedding_dim = hp.emb_dim
        else :
            print("strategy =", hp.strategy)
            print("emb_filename =", hp.emb_filename)
            self.vocab = Vocab(emb_filename=hp.emb_filename)
            self.pad_id = self.vocab.pad_id
        
            self.E = torch.nn.Embedding.from_pretrained(
                self.vocab.matrice, 
                freeze=hp.freeze_embeddings, 
                padding_idx=self.pad_id
            )
        
            embedding_dim = self.vocab.emb_dim

        # initialiser les encodings positionnels
        self.positional_encoding = T.PositionalEncoding(
            emb_size=embedding_dim,
            max_seq_length=hp.max_seq_length
        )

        # liste de blocs de tranformer
        self.blocks = nn.ModuleList([
            T.TransformerBlock(embedding_dim, self.hp.num_heads)
            for _ in range(self.hp.num_layers)
        ])

        # couche de normalisation apres les blocs de transformers 
        self.ln = nn.LayerNorm(embedding_dim)

        # couche linéaire qui map la sortie de la couche de normalisations vers la taille du vocabulaire
        self.out = nn.Linear(embedding_dim, self.vocab.vocab_size, bias=False)


    def forward(self, prefix):
        batch_size, seq_length = prefix.shape

        prefix_emb = self.E(prefix) # shape (batch_size, seq_length, emb_dim)
        x = self.positional_encoding.forward(prefix_emb) # les embeddings de tokens + les encodages positionnels

        # creer le masque causal ici 
        mask = torch.tril(
            torch.ones(seq_length, seq_length, device=prefix.device)
        ).bool()

        # Ajout des dimensions batch et nombre de têtes
        # Shape finale : (1, 1, T, T)
        # compatible avec les scores d'attention :
        # (B, num_heads, T, T)
        mask = mask.unsqueeze(0).unsqueeze(0) 
        
        for block in self.blocks :
            x = block(x, mask)

        x = self.ln(x)
        logits = self.out(x)

        # CrossEntropyLoss attend une entrée de forme :
        # (batch_size, nombre_classes, séquence)
        # donc on permute les dimensions :
        # (B, seq_length, vocab_size) -> (B, vocab_size, seq_length)
        logits = torch.swapaxes(logits, 1, 2)
        return logits 

    #-------------------------- TRAINING LOOP --------------------------
    def fit(self, train_loader, val_loader, epochs, strategy, lr=1e-3, patience=10):

        # --- INITIALISATION DE PARAMETRES ---
        LOSS_FN = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        optim = torch.optim.Adam(
            self.parameters(),
            lr=lr
        )
        best_val_loss = np.inf
        best_model_weights = None
        patience_counter = 0
        device = next(self.parameters()).device
        best_metrics = {
            "train_loss": 0.0,
            "val_loss": 0.0,
            "gap": 0.0,
            "epoch": 0
        }
        start_time = time.time()


        for epoch in range(epochs):

            # --- PHASE D'ENTRAINEMENT ---
            self.train()
            running_train_loss = 0.0
            train_tokens = 0

            for prefix, target in train_loader:
                prefix = prefix.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optim.zero_grad()
                logits = self.forward(prefix)
                loss = LOSS_FN(logits, target)
                running_train_loss += loss.item()
                loss.backward()
                optim.step()
                nb_tokens = (target != self.pad_id).sum().item()
                train_tokens += nb_tokens


            # --- PHASE DE VALIDATION ---
            self.eval()
            running_val_loss = 0.0
            val_tokens = 0

            with torch.no_grad():
                for prefix, target in val_loader:
                    prefix = prefix.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    val_logits = self.forward(prefix)
                    batch_loss = LOSS_FN(val_logits, target)
                    running_val_loss += batch_loss.item()
                    nb_tokens = (target != self.pad_id).sum().item()
                    val_tokens += nb_tokens

            # Calcul des moyennes par token
            epoch_train_loss = running_train_loss / train_tokens
            epoch_val_loss = running_val_loss / val_tokens
            gap = epoch_val_loss - epoch_train_loss

            print(
                f"Époque {epoch+1}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f}",
                flush=True
            )


            # --- GESTION DE LA PATIENCE ET SAUVEGARDE DES STATS ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(
                    self.state_dict()
                )

                best_metrics["train_loss"] = epoch_train_loss
                best_metrics["val_loss"] = epoch_val_loss
                best_metrics["gap"] = gap
                best_metrics["epoch"] = epoch + 1

                print(
                    f" --> Nouveau best metrics détecté",
                    flush=True
                )

            else:
                patience_counter += 1
                print(
                    f" --> Stagnation détectée. Patience : {patience_counter}/{patience}",
                    flush=True
                )

            if patience_counter >= patience:
                print(
                    f"\n[Early Stopping] Pas d'amélioration de la val loss depuis {patience} époques. "
                    f"Arrêt de l'entraînement à l'époque {epoch+1}.",
                    flush=True
                )
                break



        # --- FIN DES ÉPOQUES / ARRÊT : SAUVEGARDE DU MEILLEUR MODÈLE ---

        os.makedirs(
            f"best_models_strategy_{strategy}",
            exist_ok=True
        )

        emb_dim = self.hp.emb_dim
        num_layers = self.hp.num_layers
        num_heads = self.hp.num_heads
        max_seq_length = self.hp.max_seq_length

        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)
            torch.save(
                best_model_weights,
                f"best_models_strategy_{strategy}/GPT_embDim{emb_dim}_heads{num_heads}_layers{num_layers}_sqLength{max_seq_length}.pt"
            )
            print(
                "-> Fichier .pt du Best Model enregistré avec succès.",
                flush=True
            )



        # --- Fin de l'exécution de la combinaison ---
        exe_time = time.time() - start_time
        return exe_time, best_metrics

    # ------ Fonctions utilitaires ---------------
    @staticmethod
    def load(model_path, json_path):
        config = HP_GPT(json_path)
        model = MyGPT(config)
        weights = torch.load(model_path, weights_only=True, map_location='cpu')
        model.load_state_dict(weights)
        return model







