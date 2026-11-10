import os
import json


class HP_GPT:
    def __init__(self, json_file=None):

        # Architecture 
        self.num_layers = 6
        self.num_heads = 8
        self.max_seq_length = 128
        self.emb_dim = 32
        self.dropout = 0.1

        # Entraînement 
        self.minibatch = 32
        self.nb_iter = 10

        # Strategy 
        self.freeze_embeddings = True
        self.strategy = None

        # filenames
        self.train_filename = None
        self.val_filename = None
        self.emb_filename = None
        self.vocab_filename = None


        if json_file:
            self.load_and_validate(json_file)

    def _require_key(self, data, key, strategy_name=None):
        if key not in data or data[key] is None:
            if strategy_name:
                raise ValueError(f"ERREUR ({strategy_name}) : Précisez '{key}'")
            else:
                raise ValueError(f"ERREUR : Clé manquante dans le JSON : '{key}'")

    def load_and_validate(self, json_file):
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"ERREUR : Le fichier de configuration '{json_file}' est introuvable."
            )

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"ERREUR : Le fichier JSON est mal formé :\n{e}"
            )

        # Paramètres communs obligatoires
        required_keys = [
            "train_filename",
            "val_filename",
            "num_layers",
            "num_heads",
            "minibatch",
            "nb_iter",
            "strategy",
            "emb_dim",
            "emb_filename",
            "max_seq_length"
        ]

        missing = [key for key in required_keys if key not in data]
        if missing:
            raise ValueError(
                f"ERREUR : Clé(s) manquante(s) dans le JSON : {', '.join(missing)}"
            )

        # Paramètres communs
        self.train_filename = data["train_filename"]
        self.val_filename = data["val_filename"]
        self.num_layers = data["num_layers"]
        self.num_heads = data["num_heads"]
        self.minibatch = data["minibatch"]
        self.nb_iter = data["nb_iter"]
        self.strategy = data["strategy"]
        self.emb_dim = data["emb_dim"]
        self.max_seq_length = data["max_seq_length"]

        # Chargement selon la stratégie
        if self.strategy == 1:
            self.load_strategy_1(data)
        elif self.strategy == 2:
            self.load_strategy_2(data)
        elif self.strategy == 3:
            self.load_strategy_3(data)
        else:
            raise ValueError(
                f"ERREUR : stratégie inconnue : {self.strategy}. "
                "Utilisez 1, 2 ou 3."
            )

    def load_strategy_1(self, data):
        """
        Stratégie 1 :
        - vocabulaire donné
        - embeddings appris à partir de zéro
        """
        self.freeze_embeddings = False

        self._require_key(data, "vocab_filename", "stratégie 1")

        if "emb_dim" in data and data["emb_dim"] is not None:
            emb_dim = data["emb_dim"]
        else:
            raise ValueError(
                "ERREUR (stratégie 1) : Précisez 'emb_dim'"
            )

        self.vocab_filename = data["vocab_filename"]
        self.embedding_dim = emb_dim
        self.emb_dim = emb_dim

    def load_strategy_2(self, data):
        """
        Stratégie 2 :
        - embeddings pré-entraînés
        - embeddings non figés
        """
        self.freeze_embeddings = False

        self._require_key(data, "emb_filename", "stratégie 2")

        self.emb_filename = data["emb_filename"]

    def load_strategy_3(self, data):
        """
        Stratégie 3 :
        - embeddings pré-entraînés
        - embeddings figés
        """
        self.freeze_embeddings = True

        self._require_key(data, "emb_filename", "stratégie 3")

        self.emb_filename = data["emb_filename"]