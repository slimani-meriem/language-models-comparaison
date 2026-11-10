# Language Models Comparison

Projet expérimental d'étude comparative de différentes architectures de traitement du langage naturel (NLP).

## Structure du Projet

Le dépôt est organisé par architecture de modèle :

| Module | Description | Principaux composants |
| :--- | :--- | :--- |
| **`MLP/`** | Modèle perceptron multicouche | Scripts d'entraînement, fichiers de config JSON, grille de recherche `k` / dimension d'embedding |
| **`GRU/`** | Réseau récurrent à portes (Gated Recurrent Unit) | Mécanisme de cellules, génération de texte, évaluation de la perplexité|
| **`Transformer/`** | Architecture auto-attentive (type GPT) | Notebooks d'analyse des mécanismes, têtes d'attention multiples, scripts d'entraînement |

## Fonctionnalités et Pipeline

* **Entraînement et Génération** : Scripts dédiés pour entraîner chaque architecture et générer du texte.
* **Évaluation** : Calcul de la perplexité et stockage des résultats d'expérimentation (formats CSV et JSON).
* **Exploration** : Notebooks Jupyter interactifs pour comprendre le fonctionnement interne des modèles (vues d'ensemble des mécanismes).

## Données

Le corpus d'entraînement n'est **pas inclus** dans ce dépôt : je n'ai pas
l'autorisation de le redistribuer. Le dépôt contient le code, les configurations,
les résultats des expériences (CSV/JSON) et des exemples de textes générés.

Les configurations attendent l'arborescence suivante (à la racine du dépôt) :

    data/
      corpus/balises_mlp/{train,val}_12k.bpe             # MLP
      corpus/balises_gru_transformer/{train,val}_12k.bpe # GRU et Transformer
      tokenizers/bpe_tokenizer_vocab_12k.json            # vocabulaire BPE (12k)

Pour reproduire avec un autre corpus : prendre un texte français libre de droits
(par ex. Project Gutenberg ou Wikisource), entraîner un tokenizer BPE de 12 000
tokens avec la bibliothèque Hugging Face `tokenizers` (le code lit la clé
`model.vocab` du JSON), générer les fichiers train/val, puis adapter les chemins
dans les fichiers de configuration. Les résultats chiffrés ne seront pas
identiques à ceux publiés ici, qui dépendent du corpus d'origine.
```