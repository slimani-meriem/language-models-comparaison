import json
import torch
import random
import numpy as np

def update_json(file_path, key, value):
    try:
        # Si la valeur passée est le string "null" ou "None" (insensible à la casse)
        # on la transforme en vrai objet None de Python
        if isinstance(value, str) and value.lower() in ["null", "none"]:
            value = None

        with open(file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)

            data[key] = value
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
            
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable.")
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier {file_path} n'est pas un JSON valide.")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # si plusieurs gpu sont utilisés
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(model):
    k = model.hp.k
    
    def _process_file(filename, description):
        print(f"Creating {description} data from {filename}...")
        data = []
        
        # 1. On va d'abord collecter TOUS les indices du fichier dans une seule liste géante
        all_word_indices = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                tokens = line.split()
                # On convertit les mots en IDs
                word_indices = [model.vocab.get_word_index2(t) for t in tokens]
                
                # ICI LE CHANGEMENT : Au lieu de traiter la ligne tout de suite,
                # on l'ajoute à notre grand flux continu d'IDs
                all_word_indices.extend(word_indices)
                
        # 2. Une fois tout le fichier lu, on s'assure qu'on a assez de matière pour faire au moins une fenêtre
        if len(all_word_indices) <= k:
            raise ValueError(f"Erreur : Pas assez de tokens globaux dans {filename} pour la taille k={k}.")
            
        # 3. On applique la fenêtre glissante sur le flux CONTINU (les transitions se feront naturellement !)
        for i in range(len(all_word_indices) - k):
            window_indices = all_word_indices[i : i + k + 1]
            data.append(torch.tensor(window_indices, dtype=torch.long))
                    
        print(f"{len(data)} {description} examples created")
        
        if len(data) == 0:
            raise ValueError(f"Erreur : Aucun exemple créé pour le fichier {filename}.")

        return torch.stack(data)

    # Traitement du train_set
    train_dataset = _process_file(model.hp.train_filename, "training")

    # Traitement du val_set
    val_dataset = _process_file(model.hp.val_filename, "validation")

    return train_dataset, val_dataset


def prepare_data_no_overlap(model):
    k = model.hp.k
    
    def _process_file(filename, description):
        print(f"Creating {description} data from {filename}...")
        data = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                tokens = line.split()
                # On convertit les mots de la PHRASE ACTUELLE en IDs
                word_indices = [model.vocab.get_word_index2(t) for t in tokens]
                
                # S'assurer que la phrase actuelle est assez longue pour fournir au moins un exemple
                if len(word_indices) <= k:
                    # Optionnel : Tu peux décommenter la ligne suivante si tu veux ignorer les phrases trop courtes
                    # continue 
                    continue  # Ici on passe simplement à la phrase suivante sans crasher
                
                # On applique la fenêtre glissante UNIQUEMENT au sein de cette phrase
                for i in range(len(word_indices) - k):
                    window_indices = word_indices[i : i + k + 1]
                    data.append(torch.tensor(window_indices, dtype=torch.long))
                    
        print(f"{len(data)} {description} examples created")
        
        if len(data) == 0:
            raise ValueError(f"Erreur : Aucun exemple créé pour le fichier {filename}. Vérifie que les phrases ont une longueur > k={k}.")

        return torch.stack(data)

    # Traitement du train_set
    train_dataset = _process_file(model.hp.train_filename, "training")

    # Traitement du val_set
    val_dataset = _process_file(model.hp.val_filename, "validation")

    return train_dataset, val_dataset



def read_corpus(model, fichier_corpus, vocab):
    """Génère un tenseur global contenant toutes les fenêtres glissantes."""
    data = []
    k = model.hp.k
    print(f"Reading corpus with k={k}...")
    
    with open(fichier_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split(" ")
            if len(tokens) <= k:
                continue
            
            # Récupération rapide de tous les index de la ligne
            word_indices = [vocab.get_word_index2(t) for t in tokens]
            
            # Création des fenêtres glissantes
            for i in range(len(word_indices) - k):
                window_indices = word_indices[i : i + k + 1]
                data.append(torch.tensor(window_indices, dtype=torch.long))
                
    if not data:
        raise ValueError(f"Aucune donnée valide générée à partir de {fichier_corpus} avec k={k}.")
        
    return torch.stack(data)


def prepare_data_no_overlap_perplexity(model, filename):
    """
    Prépare le jeu de données pour le calcul de la perplexité (ligne par ligne)
    à partir d'un fichier donné.
    """
    k = model.hp.k
    print(f"Creating evaluation data from {filename}...")
    data = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            word_indices = [model.vocab.get_word_index2(t) for t in tokens]
            
            if len(word_indices) <= k:
                continue
            
            for i in range(len(word_indices) - k):
                window_indices = word_indices[i : i + k + 1]
                data.append(torch.tensor(window_indices, dtype=torch.long))
                
    print(f"{len(data)} evaluation examples created")
    
    if len(data) == 0:
        raise ValueError(f"Erreur : Aucun exemple créé pour le fichier {filename}. Vérifie que les phrases ont une longueur > k={k}.")

    return torch.stack(data)