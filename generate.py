import torch
import torch.nn.functional as F
import miditoolkit
import json
import gpt  

#configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = "music_vocab.json"
MODEL_PATH = "mon_nouveau_modele_musique.pth"
OUT_MIDI = "resultat_ia.mid"

#les outils
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

inverse_vocab = {v: k for k, v in vocab.items()}
tokenizer = gpt.MusicTokenizer(vocab)

#semblable a trainning
CONFIG_MUSIQUE = {
    "vocab_size": len(vocab),
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Chargement du modele
model = gpt.GPTMusicModel(CONFIG_MUSIQUE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() 


#generation de tokken
def generate(model, prompt_text, max_new_tokens=500, temperature=1.0):
    # On transforme l'amorce en chiffres
    input_ids = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
    
    for _ in range(max_new_tokens):
        #on ne garde que les 256 derniers tokens pour predire le reste (c'est sa mémoire)
        idx_cond = input_ids[:, -CONFIG_MUSIQUE["context_length"]:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            #On prend la dernière prédiction et on applique la température (= la créativité qu'on veut(haute = très créatif))
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            #on tire au sort la note suivante 
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            
            #si fin
            if inverse_vocab[next_id.item()] == "<|endoftext|>":
                break
                
    return [inverse_vocab[i.item()] for i in input_ids[0]]

#conversion texte en midi
def save_midi(tokens, output_path):
    midi = miditoolkit.midi.parser.MidiFile()
    track = miditoolkit.midi.containers.Instrument(program=0, is_drum=False, name="IA Piano")
    midi.instruments.append(track)
    
    current_tick = 0
    
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("TIME_SHIFT_"):
            current_tick += int(t.split("_")[-1]) * 10 #on multiplie par 10 car on a divisé par 10 durant la conversion
        elif t.startswith("NOTE_ON_"):
            pitch = int(t.split("_")[-1])
            #on cherche la durée associée qui suit la note
            if i + 1 < len(tokens) and "DURATION" in tokens[i+1]:
                dur_val = int(tokens[i+1].split("_")[-1])
                note = miditoolkit.Note(velocity=90, pitch=pitch, start=current_tick, end=current_tick + (dur_val * 10))
                track.notes.append(note)
                i += 1
        i += 1
    
    midi.dump(output_path)
    print(f"Musique sauvegardée dans : {output_path}")

#execution
# On donne une première note 
amorce = "NOTE_ON_60 DURATION/10_20" 
print("composition en cours")

tokens_generes = generate(model, amorce, max_new_tokens=600, temperature=1)
save_midi(tokens_generes, OUT_MIDI)
