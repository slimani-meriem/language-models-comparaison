import os
import json
import torch
import numpy as np
import sys 

class Vocab:
    def __init__(self, **kwargs):
        self.dico_voca = {}
        self.word_array = []

        # Pour stratégie 1
        if "vocab_filename" in kwargs and kwargs["vocab_filename"] not in [None, "", "null"]:
            fichier_vocab = kwargs["vocab_filename"]
            self.emb_dim = kwargs["emb_dim"]
            nb_tokens = 0
            
            # Si le fichier fourni est le .json d'origine, on le convertit d'abord en .txt
            if fichier_vocab.endswith('.json'):
                fichier_vocab_txt = fichier_vocab.replace('.json', '.txt')
                if not os.path.exists(fichier_vocab_txt):
                    self._convert_json_to_txt(fichier_vocab, fichier_vocab_txt)
                fichier_vocab = fichier_vocab_txt # On bascule sur le fichier texte pour la lecture
            
            # Lecture du fichier texte indexé
            with open(fichier_vocab, 'r', encoding='utf-8') as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    
                    
                    # Gestion du format "index mot" ou juste "mot"
                    tokens = line.split()
                    token = tokens[1] if len(tokens) > 1 else tokens[0]
                    
                    if token not in self.dico_voca:
                        self.word_array.append(token)
                        self.dico_voca[token] = nb_tokens
                        nb_tokens += 1
            
            # Sécurité : On s'assure que <unk> est présent pour get_word_index2
            if '<unk>' not in self.dico_voca:
                self.word_array.append('<unk>')
                self.dico_voca['<unk>'] = nb_tokens
                nb_tokens += 1

            # Sécurité 2 : on s'assure que <pad> est présent 
            if '<pad>' not in self.dico_voca:
                self.word_array.append('<pad>')
                self.dico_voca['<pad>'] = nb_tokens
                nb_tokens += 1

            self.vocab_size = nb_tokens
            print("vocab size (depuis fichier global) =", self.vocab_size, "emb_dim =", self.emb_dim)
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))

        # --- Structures existantes (inchangées) ---
        elif "list_filename" in kwargs:
            with open(kwargs["list_filename"], 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split(" ")
                    self.dico_voca[tokens[1]] = len(self.word_array)
                    self.word_array.append(tokens[1])
                    
        elif (
            "emb_filename" in kwargs
            and kwargs["emb_filename"] not in [None, "", "null"]
        ):
            fichier_embeddings = kwargs["emb_filename"]

            with open(fichier_embeddings, "r", encoding="utf-8") as fi:
                # Première ligne : nombre de tokens et dimension des embeddings
                ligne = fi.readline().strip()
                vocab_size_fichier, self.emb_dim = map(int, ligne.split())

                vectors = []

                for ligne in fi:
                    ligne = ligne.strip()

                    if not ligne:
                        continue

                    splitted_ligne = ligne.split()

                    token = splitted_ligne[0]
                    valeurs = splitted_ligne[1:]

                    if len(valeurs) != self.emb_dim:
                        raise ValueError(
                            f"Dimension incorrecte pour le token {token!r} : "
                            f"{len(valeurs)} valeurs trouvées, "
                            f"{self.emb_dim} attendues."
                        )

                    if token in self.dico_voca:
                        continue

                    token_id = len(self.word_array)

                    self.dico_voca[token] = token_id
                    self.word_array.append(token)

                    vector = torch.tensor(
                        [float(value) for value in valeurs],
                        dtype=torch.float32
                    )

                    vectors.append(vector)

            # Nombre de tokens réellement lus depuis le fichier d'embeddings
            nb_tokens_embeddings = len(self.word_array)

            # Ajouter <unk> s'il est absent
            if "<unk>" not in self.dico_voca:
                self.dico_voca["<unk>"] = len(self.word_array)
                self.word_array.append("<unk>")

            # Ajouter <pad> s'il est absent
            if "<pad>" not in self.dico_voca:
                self.dico_voca["<pad>"] = len(self.word_array)
                self.word_array.append("<pad>")

            # on définit les IDs
            self.vocab_size = len(self.word_array)
            self.unk_id = self.dico_voca["<unk>"]
            self.pad_id = self.dico_voca["<pad>"]

            # Matrice finale agrandie pour inclure <unk> et <pad>
            self.matrice = torch.zeros(
                (self.vocab_size, self.emb_dim),
                dtype=torch.float32
            )

            # Copier les embeddings préentraînés
            if vectors:
                self.matrice[:nb_tokens_embeddings] = torch.stack(vectors)

            # Initialiser <unk> aléatoirement seulement s'il a été ajouté
            if self.unk_id >= nb_tokens_embeddings:
                torch.nn.init.normal_(
                    self.matrice[self.unk_id],
                    mean=0.0,
                    std=0.02
                )

            # Le vecteur de <pad> reste nul
            self.matrice[self.pad_id].zero_()

            print("Taille annoncée dans le fichier :", vocab_size_fichier)
            print("Tokens réellement lus :", nb_tokens_embeddings)
            print("Taille finale du vocabulaire :", self.vocab_size)
            print("unk_id :", self.unk_id)
            print("pad_id :", self.pad_id)
            print("Forme de la matrice :", self.matrice.shape)


        self.unk_id = self.dico_voca["<unk>"]
        self.pad_id = self.dico_voca["<pad>"]

    def _convert_json_to_txt(self, json_path, txt_path):
        """ Fonction interne pour extraire et nettoyer le JSON """
        print(f"Extraction et conversion automatique de {json_path} vers {txt_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab_dict = data["model"]["vocab"]
        # Tri par ID pour garder l'ordre exact du tokenizer
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            for token, token_id in sorted_vocab:
                # Harmonisation du token inconnu pour correspondre à get_word_index2
                if token == "[UNK]":
                    token = "<unk>"
                f_out.write(f"{token_id} {token}\n")

    def get_word_index(self, mot):
        if not mot in self.dico_voca:
            print(f"ValueError : mot {mot} introuvable")
            return None
        return self.dico_voca[mot]
                

    def get_word_index2(self, mot):
        if not mot in self.dico_voca:
            sys.stderr.write("\n")
            sys.stderr.write(mot)
            return self.dico_voca['<unk>']
        return self.dico_voca[mot]
                
    def get_emb(self, mot):
        if not mot in self.dico_voca:
            return None
        return self.matrice[self.dico_voca[mot]]
    
    def get_emb_torch(self, indice_mot):
        return self.matrice[indice_mot]
        
    def get_one_hot(self, mot):
        vect = torch.zeros(len(self.dico_voca))
        vect[self.dico_voca[mot]] = 1
        return vect

    def get_word(self, index):
        if index < len(self.word_array):
            return self.word_array[index]
        return None

    def save_as_list(self, vocab_filename):
        with open(vocab_filename, "w", encoding='utf-8') as f:
            for i in range(len(self.word_array)):
                print(i, self.word_array[i], file=f)