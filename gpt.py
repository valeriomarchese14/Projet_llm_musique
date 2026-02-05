import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Tokenizer pour la Musique
class MusicTokenizer:
    def __init__(self, vocab):
        # On ajoute des tokens spéciaux pour la structure quand on connait pas
        self.str_to_int = vocab
        if "<|unk|>" not in self.str_to_int:
            self.str_to_int["<|unk|>"] = len(self.str_to_int)
        if "<|endoftext|>" not in self.str_to_int:
            self.str_to_int["<|endoftext|>"] = len(self.str_to_int)
            
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, event_list):
        #event_list est une liste de chaînes
        return [self.str_to_int.get(e, self.str_to_int["<|unk|>"]) for e in event_list]

    def decode(self, ids):
        return [self.int_to_str.get(i, "<|unk|>") for i in ids]

#Dataset pour les séquences de tokens
class MusicDataset(Dataset):
    def __init__(self, all_tokens, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #Encodage de la liste de tokens musicaux
        token_ids = tokenizer.encode(all_tokens)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def __len__(self):
        return len(self.input_ids)

#Composants du Modèle GPT
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.ReLU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
    def forward(self, x): return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -float("inf"))

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = (self.dropout(attn_weights) @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):           
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class GPTMusicModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.trf_blocks(self.drop_emb(x))
        return self.out_head(self.final_norm(x))




