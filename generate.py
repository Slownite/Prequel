import sys
from model.model_generator import generate_music
from tensorflow.keras.models import load_model
from preprocess.prepare import load_data, to_integer_base, notes_to_midi

if __name__ == "__main__":
    args = sys.args
    notes = load_data("midi_songs")
    network_input, network_output, vocab_size, pitchnames = to_integer_base(notes, 100)
    model = load_model(args[1])
    pattern = generate_music(model, network_input, vocab_size)
    if args[2] == "" or args[2] is None:
        notes_to_midi(pattern)
    else:
        notes_to_midi(pattern, args[2])
    print("generate")

