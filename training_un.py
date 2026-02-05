import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gpt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#lecture données
fichiers = ["fichier_musique1.txt"]
all_music_tokens = []
for fn in fichiers:
    with open(fn, "r", encoding="utf-8") as f:
        all_music_tokens.extend(f.read().split())

unique_tokens = sorted(list(set(all_music_tokens)))
unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(unique_tokens)}

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

# initialisation, sans charger de modele
model = gpt.GPTMusicModel(CONFIG_MUSIQUE).to(device) 

#preparation
tokenizer = gpt.MusicTokenizer(vocab)

#creation dataset
train_ds = gpt.MusicDataset(
    all_music_tokens, 
    tokenizer, 
    max_length=CONFIG_MUSIQUE["context_length"], 
    stride=128
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

#lancement


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            #calcul des prédictions
            logits = model(input_batch)
            
            #calcul de l'erreur
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), 
                target_batch.flatten()
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Époque {epoch+1}/{num_epochs} | Perte (Loss): {total_loss/len(train_loader):.4f}")
    
    #sauvegarde du modèle à la fin
    torch.save(model.state_dict(), "model_musique.pth")
    print("Entraînement terminé et modèle sauvegardé !")
    
# train_model_simple va entraîner les poids aléatoires pour qu'ils apprennent la logique de la musique
train_model_simple(model, train_loader, None, optimizer, device, num_epochs=50)

# Sauvegarde du tout nouveau modèle
torch.save(model.state_dict(), "mon_nouveau_modele_musique.pth")
