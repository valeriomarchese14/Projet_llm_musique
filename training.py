import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import gpt  

#il prendra tous les fichiers du dossier dataset_tokens2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dossier_tokens = "Projet_llm_musique/dataset_tokens2"

#prepare sur tous le dossier
fichiers = [f for f in os.listdir(dossier_tokens) if f.endswith('.txt')]
all_music_tokens = []

print(f"Chargement de {len(fichiers)} fichiers pour le vocabulaire...")
for fn in fichiers:
    chemin = os.path.join(dossier_tokens, fn)
    with open(chemin, "r", encoding="utf-8") as f:
        # On lit le texte et on le transforme en liste de mots
        tokens_du_fichier = f.read().split()
        all_music_tokens.extend(tokens_du_fichier)

#identification des tokens uniques 
unique_tokens = sorted(list(set(all_music_tokens)))

#on assure tokens spéciaux bien la 
if "<|endoftext|>" not in unique_tokens:
    unique_tokens.append("<|endoftext|>")
if "<|unk|>" not in unique_tokens:
    unique_tokens.append("<|unk|>")

vocab = {token: i for i, token in enumerate(unique_tokens)}

# Sauvegarde du vocabulaire
with open("music_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=4)

print(f"Vocabulaire créé avec {len(vocab)} tokens uniques.")

#configuration

CONFIG_MUSIQUE = {
    "vocab_size": len(vocab),
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model = gpt.GPTMusicModel(CONFIG_MUSIQUE).to(device) 
tokenizer = gpt.MusicTokenizer(vocab)

#preparation dataset
#all_music_tokens contient toute la base de données à la suite
train_ds = gpt.MusicDataset(
    all_music_tokens, 
    tokenizer, 
    max_length=CONFIG_MUSIQUE["context_length"], 
    stride=128  # Plus le stride est petit, plus le modèle voit de combinaisons
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

#entrainement
def train_model_simple(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            logits = model(input_batch)
            
            #Calcul de l'erreur
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), 
                target_batch.flatten()
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Époque {epoch+1}/{num_epochs} | Perte (Loss): {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "mon_nouveau_modele_musique.pth")
    print("Entraînement terminé et modèle sauvegardé !")

#Lancement
train_model_simple(model, train_loader, optimizer, device, num_epochs=50)
