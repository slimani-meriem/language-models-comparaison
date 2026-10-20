import os
import json


class HP_GRU:
    def __init__(self, json_file=None):
        self.num_layers = 1
        self.hidden_size = 32
        self.minibatch = 32
        self.nb_iter = 10
        self.freeze_embeddings = True

        self.strategy = None
        self.train_filename = None
        self.val_filename = None

        self.emb_filename = None
        self.vocab_filename = None

        self.emb_dim = None

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
            "h",
            "minibatch",
            "nb_iter",
            "strategy"
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
        self.hidden_size = data["h"]
        self.minibatch = data["minibatch"]
        self.nb_iter = data["nb_iter"]
        self.strategy = data["strategy"]
        self.emb_dim = data["emb_dim"]

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