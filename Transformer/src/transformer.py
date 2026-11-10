"""
    Nous allons écrire dans ce fichier le script de plusieurs classes qui constituent
    les différentes étapes de construction d'un bloc de transformer.
    Nous allons également écrire le script qui crée les encodages de position.

    Code inspiré de : https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecda

    Pour être en accord avec la littérature :
    d_model : taille  des embeddings (emb_size)
"""

import torch 
import torch.nn as nn
import math

# Etape 1 : Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_seq_length=512):
        super().__init__()

        # Initialiser un tenseur pour les position encoding
        pe = torch.zeros(max_seq_length, emb_size)

        # Création de tenseur de positions 
        # le unsqueeze est nécéssaire pour la multiplication par des div_terms plus bas
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # On va représenter le 1/1000^(2*i/d_model), présent dans le calcul du sin et cos
        # équivalent à 1000^ -(2*i/d_model)
        # avec :
        # i : varie de 0 à d_model/2 -1, on peut aussi le voir comme une variable 
        # qui varie de 0 à d_model avec un pas de 2
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() *
            - (math.log(1000.0) / emb_size)
        )

        # Commencer à la position 0 et faire des pas de 2 (2i) ==> utiliser le sinus
        pe[:, 0::2] = torch.sin(position * div_term)

        # Commencer à la position 1 et faire des pas de 2 (2i+1) ==> utiliser le cosinus
        pe[:, 1::2] = torch.cos(position * div_term)

        # Enregistrer les encodages de positions en tant que buffer 
        # (ne sera pas considéré comme un paramètre du modèle)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_seq_length, emb_size)
    
        
    def forward(self, x) :
        # x : Shape (batch_size, sequence_length, emb_size)
        # Ajouter l’encodage positionnel correspondant à la longueur réelle de la séquence.
        return x + self.pe[:, :x.size(1)]


# Etape 2 : Multi-head attention
class MultiHeadAttention(nn.Module) :
    def __init__(self, emb_size, num_heads, qkv_bias=False):
        super().__init__()
        assert emb_size % num_heads == 0

        self.model_dim = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        # Création des matrice de paramètres W_q, W_k et W_v
        # ce sont des matrices de dimensions (d_model x d_model)
        self.query = nn.Linear(self.model_dim, self.model_dim, bias=qkv_bias)
        self.key = nn.Linear(self.model_dim, self.model_dim, bias=qkv_bias)
        self.value = nn.Linear(self.model_dim, self.model_dim, bias=qkv_bias)

        # La couche de sortie, à appliquer après avoir 
        # concaténé les sorties de toutes les têtes d'attention
        self.out = nn.Linear(self.model_dim, self.model_dim, bias=False)

    def forward(self, x, mask=None) :
        # x : Shape (batch_size, seq_length, model_dim)
        batch_size, seq_length, _ = x.shape

        # 1. Créer les matrices Q, K et V 
        # Shape = shape de x = (batch_size, seq_length, model_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. Séparer en plusieurs têtes 
        # Q, K et V seront : Shape (batch_size, seq_length, num_heads, model_dim)
        # Pour la suite des calculs on veut : Shape (batch_size, num_heads, seq_length, model_dim)
        Q = Q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # 3. Calculer les scores 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 4. appliquer le masque causal qui est passé en paramètre de la méthode
        # Il servira à masquer le futur 
        # Les scores mis à -inf se transformeront en 0 apres le softmax
        if mask != None :
            scores = scores.masked_fill(mask==0, float('-inf'))

        # 5. matrice d'attention
        attention = torch.softmax(scores, dim=-1)

        # 6. Calculer le Y = AV
        Y = torch.matmul(attention, V)

        # 7. Concatener les resultats des différentes têtes 
        # On revient à la shape des x : (batch_size, seq_length, model_dim)
        # on utilise .contiguous() pour réellement organiser les données selon 
        # les nouvelles dimensions imposé par .transpose() juste avant
        Y = Y.transpose(1,2).contiguous().view(batch_size, seq_length, self.model_dim)

        return self.out(Y)



# 4. Layer Normalization (On utilisera cependant nn.LayerNorm par la suite)
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.alpha * norm_x + self.beta


# 5. Feed Forward Network Layer
class FeedForward(nn.Module):
  def __init__(self, emb_size):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(emb_size, emb_size*4),
        nn.ReLU(),
        nn.Linear(emb_size*4, emb_size)
      )

  def forward(self, x):
    return self.network(x)


# Construction du bloc de transformer 
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1, qkv_bias=False):
        super().__init__()

        # couche multi attention
        self.mha = MultiHeadAttention(
            emb_size=emb_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )

        # couche feed forward
        self.ff = FeedForward(emb_size=emb_size)

        # les 2 couches de normalisation (une après le mha et l'autre après le ff)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        # drop-out layer (section 5.4 de Attention Is All You Need)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None) :
        mha_output = self.ln1(
            x + 
            self.dropout(self.mha(x, mask))
        )

        ff_output = self.ln2(
            mha_output +
            self.dropout(self.ff(mha_output))
        )

        return ff_output

