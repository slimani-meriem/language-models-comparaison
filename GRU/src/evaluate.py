import argparse
import torch
import torch.nn.functional as F
from gru import GRU
from hp import HP_GRU

"""
Évaluation autorégressive d'un modèle GRU sur un texte à trous.
Le contexte gauche sert d'amorce, puis chaque candidat est testé
en parallèle (vectorisé) pour prédire le contexte droit.

    python evaluate.py \
        --mask_file [1]  \
        --candidates_file [2] \
        --model_file [3]\
        --config_file [4] \
        --top_k [5]
"""


def get_candidates(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def process_sentence(line, vocab, candidates, device, model, top_k=10):
    line_parts = line.split()

    if not line_parts:
        return None

    # Extraction du mot masqué
    masked_token_index = int(line_parts[0])
    sentence_tokens = line_parts[1:]

    if masked_token_index >= len(sentence_tokens):
        return None
    masked_token = sentence_tokens[masked_token_index]

    # Extraire le contexte droit et gauche
    left_tokens = sentence_tokens[:masked_token_index]
    right_tokens = sentence_tokens[masked_token_index + 1 :]

    left_ids = []
    right_ids = []

    for token in left_tokens:
        token_id = vocab.get_word_index2(token)
        if token_id is None:
            return None
        left_ids.append(int(token_id))

    for token in right_tokens:
        token_id = vocab.get_word_index2(token)
        if token_id is None:
            return None
        right_ids.append(int(token_id))

    # Tenseur du contexte gauche : shape (1, seq_len_left)
    left_tensor = torch.tensor(
        [left_ids],
        dtype=torch.long,
        device=device,
    )

    # 1. Gestion des candidats locaux sans écraser d'élément
    local_candidates = list(candidates)
    if masked_token not in local_candidates:
        local_candidates.append(masked_token)

    # 2. Construction des candidats sous forme d'IDs uniques
    candidate_ids = []
    candidate_tokens = []

    for token in local_candidates:
        t_id = vocab.get_word_index2(token)
        if t_id is not None and t_id not in candidate_ids:
            candidate_ids.append(int(t_id))
            candidate_tokens.append(token)

    if not candidate_ids:
        return None

    num_candidates = len(candidate_ids)

    with torch.inference_mode():
        # 3. Traitement unique du contexte gauche
        logits_left, hidden_left = model.forward_evaluate(left_tensor)

        # Dernier logit du contexte gauche -> P(candidat | contexte gauche)
        masked_logits = logits_left[0, -1, :]
        log_prob_masked = F.log_softmax(masked_logits, dim=-1)

        # Extraction des log-probs initiales pour tous les candidats (Shape: [num_candidates])
        candidate_ids_tensor = torch.tensor(
            candidate_ids, dtype=torch.long, device=device
        )
        scores = log_prob_masked[candidate_ids_tensor].clone()

        # 4. VECTORISATION DU CONTEXTE DROIT (BATCHING)
        # current_tokens shape : (num_candidates, 1)
        current_tokens = candidate_ids_tensor.unsqueeze(1)

        # Duplication de l'état caché pour tout le batch de candidats
        # Adapte selon la forme de ton hidden state: (num_layers, batch_size, hidden_dim)
        if hidden_left.dim() == 3:
            hidden = hidden_left.repeat(1, num_candidates, 1)
        else:
            hidden = hidden_left.repeat(num_candidates, 1)

        # Évaluation parallèle du contexte droit
        for right_target_id in right_ids:
            logits, hidden = model.forward_evaluate(current_tokens, hidden)

            # logits shape : (num_candidates, 1, vocab_size)
            next_token_logits = logits[:, -1, :]
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            # On ajoute la log-prob du vrai mot droit pour chaque candidat
            scores += next_token_log_probs[:, right_target_id]

            # Le prochain token injecté est le vrai mot observé (dupliqué pour le batch)
            current_tokens = torch.full(
                (num_candidates, 1),
                right_target_id,
                dtype=torch.long,
                device=device,
            )

    # 5. Tri des résultats
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

    # 6. Affichage des résultats
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
        description="Évaluation du modèle GRU sur un fichier à trous"
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
        help="Chemin du fichier des tokens candidats",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Chemin du modèle entraîné",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Chemin du fichier de configuration JSON",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Valeur d'affichage pour le Top-K",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    # Charger le modèle
    model = GRU.load(model_path=args.model_file, json_path=args.config_file)
    model = model.to(device)
    model.eval()

    # Charger les candidats
    candidates = get_candidates(args.candidates_file)

    print("Modèle chargé avec succès.")
    print("Taille du vocabulaire :", model.vocab.vocab_size)
    print("Nombre de candidats chargés :", len(candidates))
    print("Lecture du fichier :", args.mask_file)

    total_score = 0.0
    evaluated = 0
    skipped = 0

    with open(args.mask_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            print("\nTraitement de la ligne", line_number)

            score = process_sentence(
                line=line,
                vocab=model.vocab,
                candidates=candidates,
                device=device,
                model=model,
                top_k=args.top_k,
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
    print("Phrases évaluées :", evaluated)
    print("Phrases ignorées :", skipped)

    if evaluated > 0:
        mean_score = total_score / evaluated
        print(
            f"Score Moyen de Rang (1.0 = 1er, 0.01 = 100ème) :"
            f" {mean_score:.4f}"
        )
    else:
        print("Aucune phrase n'a pu être évaluée.")


if __name__ == "__main__":
    main()











