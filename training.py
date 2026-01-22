import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gpt  # Ton fichier contenant GPTMusicModel

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lecture des données pour construire le vocabulaire
fichiers = ["fichier_musique1.txt"]
all_music_tokens = []
for fn in fichiers:
    with open(fn, "r", encoding="utf-8") as f:
        all_music_tokens.extend(f.read().split())

unique_tokens = sorted(list(set(all_music_tokens)))
unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(unique_tokens)}

# Configuration du modèle
CONFIG_MUSIQUE = {
    "vocab_size": len(vocab),
    "context_length": 256, # Nombre de tokens passés en une fois
    "emb_dim": 384,        # Dimension des vecteurs
    "n_heads": 8,          # Nombre de têtes d'attention
    "n_layers": 6,         # Nombre de blocs Transformer
    "drop_rate": 0.1,
    "qkv_bias": False
}

# --- INITIALISATION DU MODÈLE DE ZÉRO ---
# On ne charge aucun .pth ici
model = gpt.GPTMusicModel(CONFIG_MUSIQUE).to(device) 

# --- PRÉPARATION DU TRAINING ---
tokenizer = gpt.MusicTokenizer(vocab)

# 2. Création du Dataset (on utilise le préfixe gpt. et les bons arguments)
train_ds = gpt.MusicDataset(
    all_music_tokens, 
    tokenizer, 
    max_length=CONFIG_MUSIQUE["context_length"], 
    stride=128
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# --- LANCEMENT ---


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            # Déplacement des données sur le GPU ou CPU
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            # Calcul des prédictions
            logits = model(input_batch)
            
            # Calcul de l'erreur (Loss)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), 
                target_batch.flatten()
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Époque {epoch+1}/{num_epochs} | Perte (Loss): {total_loss/len(train_loader):.4f}")
    
    # Sauvegarde du modèle à la fin
    torch.save(model.state_dict(), "model_musique.pth")
    print("Entraînement terminé et modèle sauvegardé !")
    
# train_model_simple va entraîner les poids aléatoires pour qu'ils 
# apprennent la logique de ta musique.
train_model_simple(model, train_loader, None, optimizer, device, num_epochs=50)

# Sauvegarde du tout nouveau modèle
torch.save(model.state_dict(), "mon_nouveau_modele_musique.pth")