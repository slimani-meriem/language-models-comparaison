import time
import torch 
import torch.nn as nn
from Vocab import Vocab
from hp import HP_GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import copy 
import os


class GRU(nn.Module):
    def __init__(self, hp):
        super(GRU, self).__init__()
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

        self.gru = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=hp.hidden_size, 
            num_layers=hp.num_layers,
            bias=True, 
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hp.hidden_size, self.vocab.vocab_size)

        print("Taille du vocabulaire :", self.vocab.vocab_size)
        print("Indice de <pad> :", self.pad_id)
        print("Indice de <unk> :", self.vocab.unk_id)
        print("Taille embeddings :", self.E.weight.shape)
    
    def forward(self, x, lengths):
        # x : (batch_size, seq_len)
        print(x.shape)

        x = self.E(x)
        # x : (batch_size, seq_len, embedding_dim)

        x_packed = pack_padded_sequence(  # on pack selon les vraies longueurs de préfixes car ls prédiction se feront selon ça
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        #print("Vraies longueurs des préfixes : ", lengths)
        #print("Shape x packed (prefixe packed) :", x_packed.data.shape)
        output_packed, h_n = self.gru(x_packed)
        #print("Shape output_packed :", output_packed.data.shape)
        #print("Shape h_n :", h_n.shape)
        output, _ = pad_packed_sequence(
            output_packed,
            batch_first=True
        )

        #print("Shape output :", output.shape)
        #print("\n" + "="*70)
        #print("--- INSPECTION COMPLÈTE DE OUTPUT ---")
        #print("Taille globale :", output.shape) # (batch_size, seq_len, hidden_dim)

        logits = self.fc(output) # h_n, dim output 
        #print("Shape logits :", logits.shape)
        #print("--------------------------------------------------------------")
        return logits, h_n

    def forward_evaluate(self, x, h_0=None):
        """
        Forward utilisé pendant l'évaluation mot par mot.

        x :
            [batch_size, sequence_length]

        h_0 :
            vecteur d'état précédent ;
            [num_layers, batch_size, hidden_size]

        Retourne :
            logits : [batch_size, sequence_length, vocab_size]
            h_n    : nouveau vecteur d'état
        """

        embeddings = self.E(x)

        output, h_n = self.gru(
            embeddings,
            h_0
        )

        logits = self.fc(output)

        return logits, h_n
    


    # ------ Fonctions utilitaires ---------------
    @staticmethod
    def load(model_path, json_path):
        config = HP_GRU(json_path)
        model = GRU(config)
        weights = torch.load(model_path, weights_only=True, map_location='cpu')
        model.load_state_dict(weights)
        return model



    def fit(self, train_loader, val_loader, epochs, strategy, lr=1e-3, patience=10):

        # --- INITIALISATION DE PARAMETRES ---
        LOSS_FN = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        optim = torch.optim.Adam(self.parameters(), lr=lr)
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

        for epoch in range(epochs) :
            # --- PHASE D'ENTRAINEMENT ---
            self.train()
            running_train_loss = 0.0
            train_tokens = 0

            for batch, lengths in train_loader :
                batch = batch.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                #print("shape du batch :", batch.shape)
                #print("Vraies longueurs de sphrases :", lengths)
                prefix = batch[:, :-1]
                #print("shape des prefixes :", prefix.shape)
                target = batch[:, 1:]
                #print("shape des targets :", target.shape)
                prefix_lengths = lengths - 1

                optim.zero_grad()
                logits, _ = self.forward(prefix, prefix_lengths)
                loss = LOSS_FN(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1)
                )
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
                for batch, lengths in val_loader:
                    # --- ENVOI DU BATCH SUR LE GPU ---
                    batch = batch.to(device)

                    prefix = batch[:, :-1]
                    target = batch[:, 1:]
                    prefix_lengths = lengths - 1

                    val_logits, _ = self.forward(prefix, prefix_lengths)
                    batch_loss = LOSS_FN(
                        val_logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1)
                    )

                    running_val_loss += batch_loss.item()
                    nb_tokens = (target != self.pad_id).sum().item()
                    val_tokens += nb_tokens


            # Calcul des moyennes par batch sur toute l'époque
            epoch_train_loss = running_train_loss / train_tokens
            epoch_val_loss = running_val_loss / val_tokens
            gap = epoch_val_loss - epoch_train_loss

            print(f"Époque {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}", flush=True)

            # --- GESTION DE LA PATIENCE ET SAUVEGARDE DES STATS ---
            if epoch_val_loss < best_val_loss :
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                # Deep copy des poids
                best_model_weights = copy.deepcopy(self.state_dict())
                
                best_metrics["train_loss"] = epoch_train_loss
                best_metrics["val_loss"] = epoch_val_loss
                best_metrics["gap"] = gap
                best_metrics["epoch"] = epoch + 1
                print(f" --> Nouveau best metrics détecté", flush=True)
            else:
                patience_counter += 1
                print(f" --> Stagnation détectée. Patience : {patience_counter}/{patience}", flush=True)
       
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Pas d'amélioration de la val loss depuis {patience} époques. Arrêt de l'entraînement à l'époque {epoch+1}.", flush=True)
                break

        # --- FIN DES ÉPOQUES / ARRÊT : SAUVEGARDE DU MEILLEUR MODÈLE ---
        os.makedirs(f"best_models_strategy_{strategy}", exist_ok=True)
        B = self.hp.minibatch
        h = self.hp.hidden_size
        emb_dim = self.hp.emb_dim

        
        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)
            torch.save(best_model_weights, f"best_models_strategy_{strategy}/GRU_h{h}_embDim{emb_dim}.pt")
            print("-> Fichier .pt du Best Model enregistré avec succès.", flush=True)
        
        

        # --- Fin de l'exécution de la combinaison ---
        exe_time = time.time() - start_time

        return exe_time, best_metrics
