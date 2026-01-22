import miditoolkit

nomFichier="FichiersMidi/GrandMidiPiano/Bach, Johann Sebastian, Partita in G major, BWV 829, aecenXn3obw.mid"
midi = miditoolkit.MidiFile(nomFichier)

tokens = []

for inst in midi.instruments:
    for note in inst.notes:
        tokens.append(f"POSITION_{note.start}")
        tokens.append(f"NOTE_ON_{note.pitch}")
        tokens.append(f"DURATION_{note.end - note.start}")

for token in tokens:
    print(token)
