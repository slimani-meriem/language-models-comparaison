"""
python evaluate_all_vocab.py \
    --mask_file [1] \
    --model_file [2] \
    --config_file [3] \
    --top_k [4] \
    --batch_size [5]  
"""

import argparse
import torch
import torch.nn.functional as F
from gru import GRU


def process_sentence_vocab(
    line,
    vocab,
    id_to_token,
    device,
    model,
    top_k=10,
    batch_size=512,
):
    line_parts = line.split()

    if not line_parts:
        return None

    masked_token_index = int(line_parts[0])
    sentence_tokens = line_parts[1:]

    if masked_token_index >= len(sentence_tokens):
        return None
    masked_token = sentence_tokens[masked_token_index]

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

    target_id = vocab.get_word_index2(masked_token)
    if target_id is None:
        return None
    target_id = int(target_id)

    vocab_size = vocab.vocab_size
    all_vocab_ids = torch.arange(vocab_size, device=device, dtype=torch.long)

    left_tensor = torch.tensor(
        [left_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        # 1. Traitement du contexte gauche
        logits_left, hidden_left = model.forward_evaluate(left_tensor)

        # Log-probs sur tout le vocabulaire pour la position masquée
        masked_logits = logits_left[0, -1, :]
        base_log_probs = F.log_softmax(masked_logits, dim=-1)  # Shape: (vocab_size,)

        final_scores = torch.zeros(vocab_size, device=device)

        # 2. Vectorisation par lots (batchs) sur tout le vocabulaire
        for start in range(0, vocab_size, batch_size):
            end = min(start + batch_size, vocab_size)
            batch_candidate_ids = all_vocab_ids[start:end]
            curr_batch_size = len(batch_candidate_ids)

            # Scores initiaux pour ce batch
            batch_scores = base_log_probs[start:end].clone()

            # Duplication de l'état caché
            if hidden_left.dim() == 3:
                # Shape: (num_layers, curr_batch_size, hidden_dim)
                hidden = hidden_left.repeat(1, curr_batch_size, 1)
            else:
                hidden = hidden_left.repeat(curr_batch_size, 1)

            current_tokens = batch_candidate_ids.unsqueeze(1)  # Shape: (curr_batch_size, 1)

            # Évaluation autorégressive du contexte droit
            for right_target_id in right_ids:
                logits, hidden = model.forward_evaluate(current_tokens, hidden)

                next_token_logits = logits[:, -1, :]
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

                batch_scores += next_token_log_probs[:, right_target_id]

                current_tokens = torch.full(
                    (curr_batch_size, 1),
                    right_target_id,
                    dtype=torch.long,
                    device=device,
                )

            final_scores[start:end] = batch_scores

    # 3. Rang du mot attendu sur l'ensemble du vocabulaire
    target_score = final_scores[target_id].item()
    rank = (final_scores > target_score).sum().item() + 1

    # 4. Top-K
    top_scores, top_indices = torch.topk(final_scores, top_k)
    top_scores = top_scores.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    print("\n" + "=" * 60)
    print(f"Mot attendu : '{masked_token}' (ID: {target_id})")
    print(f"Rang du mot attendu : {rank}/{vocab_size}")

    print(f"\nTop-{top_k} du vocabulaire complet :")
    for r, (idx, sc) in enumerate(zip(top_indices, top_scores), start=1):
        tok_str = id_to_token.get(idx, f"<id_{idx}>")
        print(f"{r}. {tok_str:<15} | id = {idx:<6} | log-prob = {sc:.4f}")

    return rank, vocab_size


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation globale du GRU sur l'ensemble du vocabulaire"
    )

    parser.add_argument(
        "--mask_file",
        type=str,
        required=True,
        help="Chemin du fichier .mask",
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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Taille du batch de candidates à passer sur GPU",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    # Charger le modèle
    model = GRU.load(model_path=args.model_file, json_path=args.config_file)
    model = model.to(device)
    model.eval()

    vocab = model.vocab
    # Reconstruction de la table inverse d'indexation
    id_to_token = {v: k for k, v in vocab.dico_voca.items()} if hasattr(vocab, "dico_voca") else {}

    print("Modèle chargé avec succès.")
    print("Taille du vocabulaire :", vocab.vocab_size)
    print("Lecture du fichier :", args.mask_file)

    ranks = []
    evaluated = 0
    skipped = 0

    with open(args.mask_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            print("\nTraitement de la ligne", line_number)

            res = process_sentence_vocab(
                line=line,
                vocab=vocab,
                id_to_token=id_to_token,
                device=device,
                model=model,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )

            if res is None:
                skipped += 1
                print("Ligne ignorée.")
                continue

            rank, v_size = res
            evaluated += 1
            ranks.append((rank, v_size))

    print("\n" + "=" * 50)
    print("Résultats de l'évaluation globale GRU (Vocabulaire complet)")
    print("=" * 50)
    print("Phrases évaluées :", evaluated)
    print("Phrases ignorées  :", skipped)

    if evaluated > 0:
        mean_rank = sum(r for r, _ in ranks) / evaluated
        mrr = sum(1.0 / r for r, _ in ranks) / evaluated
        normalized_score = sum((v - r + 1) / v for r, v in ranks) / evaluated

        top1 = sum(1 for r, _ in ranks if r <= 1) / evaluated * 100
        top5 = sum(1 for r, _ in ranks if r <= 5) / evaluated * 100
        top10 = sum(1 for r, _ in ranks if r <= 10) / evaluated * 100

        print(f"Rang moyen (Mean Rank)           : {mean_rank:.2f}")
        print(f"Score de Rang Normalisé          : {normalized_score:.4f}")
        print(f"Précision Top-1                  : {top1:.2f}%")
        print(f"Précision Top-5                  : {top5:.2f}%")
        print(f"Précision Top-10                 : {top10:.2f}%")
    else:
        print("Aucune phrase n'a pu être évaluée.")


if __name__ == "__main__":
    main()