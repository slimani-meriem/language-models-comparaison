"""
    Ce script évalue la capacité d'un modèle de langage MLP à prédire un mot masqué 
    dans un texte à trous (cloze task). Il prend une fenêtre de taille $2k+1$ 
    autour du trou, substitue la position masquée par chaque mot d'une liste de 
    candidats, puis calcule la somme des log-probabilités conditionnelles sur 
    l'ensemble des fenêtres de taille $k$. 
    Il affiche enfin le rang du mot cible ainsi qu'un score moyen normalisé.

    python evaluate.py \
        --mask_file [1] \
        --candidates_file [2] \
        --model_file [3] \
        --config_file [4] \
        --top_k [5] \
        --batch_size [6]

    [1] : Chemin vers le fichier .mask contenant le texte à trous.
    [2] : Chemin vers le fichier texte contenant la liste des mots candidats à tester.
    [3] : Chemin vers le fichier modèle (.pt).
    [4] : Chemin vers le fichier de configuration JSON des hyperparamètres du modèle.
    [5] : (Optionnel, défaut: 10) Nombre de meilleurs candidats à afficher dans le log pour chaque phrase.
    [6] : (Optionnel, défaut: 256) Nombre de candidats traités en parallèle dans un lot lors de la vectorisation du contexte droit.
"""




import argparse
import torch
import torch.nn.functional as F
from mlp import MLP, Hp



def resize_model(model, state_dict):
    """Adapte les dimensions du modèle aux dimensions des poids sauvegardés."""
    checkpoint_vocab_size = state_dict["E.weight"].shape[0]
    embedding_dim = state_dict["E.weight"].shape[1]

    if model.E.weight.shape != state_dict["E.weight"].shape:
        model.E = torch.nn.Embedding(
            checkpoint_vocab_size,
            embedding_dim,
        )

    if model.W2.shape != state_dict["W2"].shape:
        model.W2 = torch.nn.Parameter(
            torch.empty_like(state_dict["W2"])
        )

    if model.b2.shape != state_dict["b2"].shape:
        model.b2 = torch.nn.Parameter(
            torch.empty_like(state_dict["b2"])
        )

    return checkpoint_vocab_size


def get_candidates(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def process_sentence(
    line,
    vocab,
    candidates,
    model,
    device,
    k,
    top_k=10,
    batch_size=256,
):
    parts = line.split()

    if len(parts) < 2:
        return None

    try:
        masked_index = int(parts[0])
    except ValueError:
        return None

    tokens = parts[1:]

    if not 0 <= masked_index < len(tokens):
        return None

    if masked_index < k or masked_index + k >= len(tokens):
        return None

    masked_token = tokens[masked_index]

    if masked_token not in vocab:
        return None

    unk_id = vocab.get("<unk>")

    # Extraire la fenêtre de taille 2k + 1 autour du trou
    local_tokens = tokens[
        masked_index - k : masked_index + k + 1
    ]

    local_ids = []
    for token in local_tokens:
        token_id = vocab.get(token, unk_id)
        if token_id is None:
            return None
        local_ids.append(int(token_id))

    local_tensor = torch.tensor(
        local_ids,
        dtype=torch.long,
        device=device,
    )

    # 1. Gestion des candidats locaux sans altérer la liste globale
    local_candidates = list(candidates)
    if masked_token not in local_candidates:
        local_candidates[0] = masked_token

    # 2. Filtrage des candidats valides dans le vocabulaire
    candidate_ids = []
    candidate_tokens = []

    for token in local_candidates:
        if token in vocab:
            t_id = int(vocab[token])
            if t_id not in candidate_ids:
                candidate_ids.append(t_id)
                candidate_tokens.append(token)

    if not candidate_ids:
        return None

    candidate_ids_tensor = torch.tensor(
        candidate_ids, dtype=torch.long, device=device
    )
    number_of_candidates = len(candidate_ids)

    with torch.inference_mode():
        # --------------------------------------------------
        # 3. Prédiction basée sur le contexte gauche
        # P(candidat | k mots à gauche)
        # --------------------------------------------------
        left_context = local_tensor[:k].unsqueeze(0)
        logits = model(left_context)

        if isinstance(logits, tuple):
            logits = logits[0]

        log_probabilities = F.log_softmax(logits, dim=-1)[0]
        scores = log_probabilities[candidate_ids_tensor].clone()

        # --------------------------------------------------
        # 4. Probabilités cumulées sur les k mots suivants à droite
        # --------------------------------------------------
        for shift in range(1, k + 1):
            right_target_id = local_ids[k + shift]
            context_template = local_tensor[shift : shift + k]
            candidate_position = k - shift

            for start in range(0, number_of_candidates, batch_size):
                end = min(start + batch_size, number_of_candidates)
                current_candidates = candidate_ids_tensor[start:end]

                contexts = context_template.unsqueeze(0).repeat(
                    len(current_candidates), 1
                )
                contexts[:, candidate_position] = current_candidates

                logits = model(contexts)
                if isinstance(logits, tuple):
                    logits = logits[0]

                log_probabilities = F.log_softmax(logits, dim=-1)
                scores[start:end] += log_probabilities[:, right_target_id]

    # 5. Classement des candidats
    scores_list = scores.cpu().tolist()
    results = [
        {"id": c_id, "token": tok, "log-prob": sc}
        for c_id, tok, sc in zip(candidate_ids, candidate_tokens, scores_list)
    ]
    results.sort(key=lambda x: x["log-prob"], reverse=True)

    N = len(results)

    # Calcul du rang du mot attendu
    rank = None
    for idx, item in enumerate(results, start=1):
        if item["token"] == masked_token:
            rank = idx
            break

    score = (N - rank + 1) / N if rank is not None else 0.0

    # 6. Impression des résultats pour la phrase
    print("\n" + "=" * 60)
    print(f"Mot attendu : '{masked_token}'")
    if rank:
        print(f"Rang du mot attendu : {rank}/{N} | Score : {score:.4f}")
    else:
        print(f"Mot attendu non trouvé parmi les {N} candidats.")

    print(f"\nTop-{top_k} final :")
    for r, item in enumerate(results[:top_k], start=1):
        print(
            f"{r}. {item['token']} "
            f"| id = {item['id']} "
            f"| log-prob = {item['log-prob']:.4f}"
        )

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation du MLP sur un corpus à trous avec ensemble de candidats"
    )

    parser.add_argument(
        "--mask_file",
        type=str,
        required=True,
        help="Chemin du fichier .mask",
    )

    parser.add_argument(
        "--candidates_file",
        type=str,
        required=True,
        help="Chemin du fichier des candidats",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Chemin du modèle sauvegardé",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Chemin de la configuration du modèle",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Taille du Top-K affiché",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Taille du lot pour le traitement des candidats",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # Charger la configuration et instancier le modèle
    hp = Hp(args.config_file)
    model = MLP(hp)

    state_dict = torch.load(args.model_file, map_location="cpu")
    output_vocab_size = resize_model(model, state_dict)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    vocab = model.vocab.dico_voca
    k = int(hp.k)

    candidates = get_candidates(args.candidates_file)

    print(f"Taille du dictionnaire : {len(vocab)}")
    print(f"Taille de sortie du modèle : {output_vocab_size}")
    print(f"Nombre de candidats chargés : {len(candidates)}")
    print(f"Taille du contexte (k) : {k}")
    print(f"Évaluation du fichier : {args.mask_file}")

    total_score = 0.0
    evaluated = 0
    skipped = 0

    with open(args.mask_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            print(f"\nTraitement de la ligne {line_number}")

            score = process_sentence(
                line=line,
                vocab=vocab,
                candidates=candidates,
                model=model,
                device=device,
                k=k,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )

            if score is None:
                skipped += 1
                print("Ligne ignorée.")
                continue

            evaluated += 1
            total_score += score

    print("\n" + "=" * 50)
    print("Résultats de l'évaluation")
    print("=" * 50)
    print(f"Phrases évaluées : {evaluated}")
    print(f"Phrases ignorées  : {skipped}")

    if evaluated > 0:
        mean_score = total_score / evaluated
        print(f"Score Moyen de Rang : {mean_score:.4f}")
    else:
        print("Aucune phrase n'a pu être évaluée.")


if __name__ == "__main__":
    main()