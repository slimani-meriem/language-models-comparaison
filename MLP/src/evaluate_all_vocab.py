"""
    Ce script évalue de façon exhaustive le modèle de langage MLP sur 
    l'ensemble du vocabulaire pour prédire le mot masqué dans un texte à 
    trous (cloze task). Pour chaque phrase, il calcule la log-probabilité 
    combinée sur une fenêtre de taille $2k+1$ en faisant passer successivement 
    tous les mots du vocabulaire dans la position masquée. Il génère ensuite le 
    rang exact du mot attendu, le score de rang normalisé ainsi que les métriques 
    de précision Top-1, Top-5 et Top-10.

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


def process_sentence(
    line,
    vocab,
    id_to_token,
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

    target_id = int(vocab[masked_token])
    unk_id = vocab.get("<unk>")

    # Extraire la fenêtre de taille 2k + 1 autour du trou
    local_tokens = tokens[masked_index - k : masked_index + k + 1]

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

    vocab_size = len(vocab)
    all_vocab_ids = torch.arange(vocab_size, device=device, dtype=torch.long)

    with torch.inference_mode():
        # 1. Score issu du contexte gauche : P(vocab_i | k mots à gauche)
        left_context = local_tensor[:k].unsqueeze(0)
        logits = model(left_context)

        # Log-probabilités pour l'ensemble du vocabulaire
        scores = F.log_softmax(logits, dim=-1)[0]  # Shape: (vocab_size,)

        # 2. Accumulation des probabilités sur les k mots suivants à droite
        for shift in range(1, k + 1):
            right_target_id = local_ids[k + shift]
            context_template = local_tensor[shift : shift + k]
            candidate_position = k - shift

            # batching sur l'ensemble du vocabulaire
            for start in range(0, vocab_size, batch_size):
                end = min(start + batch_size, vocab_size)
                current_batch_ids = all_vocab_ids[start:end]
                current_batch_len = len(current_batch_ids)

                contexts = context_template.unsqueeze(0).repeat(current_batch_len, 1)
                contexts[:, candidate_position] = current_batch_ids

                logits_right = model(contexts)
                if isinstance(logits_right, tuple):
                    logits_right = logits_right[0]

                log_probs_right = F.log_softmax(logits_right, dim=-1)
                scores[start:end] += log_probs_right[:, right_target_id]

    # 3. Calcul du rang sur l'ensemble du vocabulaire
    target_score = scores[target_id].item()
    # Le rang est le nombre de mots avec un score strictement supérieur + 1
    rank = (scores > target_score).sum().item() + 1

    # 4. Extraction du Top-K
    top_scores, top_indices = torch.topk(scores, top_k)
    top_scores = top_scores.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    # Affichage des logs pour la phrase
    print("\n" + "=" * 60)
    print(f"Mot attendu : '{masked_token}' (ID: {target_id})")
    print(f"Rang du mot attendu sur tout le vocabulaire : {rank}/{vocab_size}")

    print(f"\nTop-{top_k} du vocabulaire complet :")
    for r, (idx, sc) in enumerate(zip(top_indices, top_scores), start=1):
        tok_str = id_to_token.get(idx, f"<id_{idx}>")
        print(f"{r}. {tok_str:<15} | id = {idx:<6} | log-prob = {sc:.4f}")

    return rank, vocab_size


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation globale du MLP sur tout le vocabulaire"
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
        help="Taille du Top-K affiché dans les logs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Taille de lot pour le passage avant sur l'ensemble du vocabulaire",
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
    id_to_token = {v: k for k, v in vocab.items()}
    k = int(hp.k)

    print(f"Taille du dictionnaire : {len(vocab)}")
    print(f"Taille de sortie du modèle : {output_vocab_size}")
    print(f"Taille du contexte (k) : {k}")
    print(f"Évaluation du fichier : {args.mask_file}")

    ranks = []
    evaluated = 0
    skipped = 0

    with open(args.mask_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            print(f"\nTraitement de la ligne {line_number}")

            res = process_sentence(
                line=line,
                vocab=vocab,
                id_to_token=id_to_token,
                model=model,
                device=device,
                k=k,
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
    print("Résultats de l'évaluation globale (Vocabulaire complet)")
    print("=" * 50)
    print(f"Phrases évaluées : {evaluated}")
    print(f"Phrases ignorées  : {skipped}")

    if evaluated > 0:
        mean_rank = sum(r for r, _ in ranks) / evaluated
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