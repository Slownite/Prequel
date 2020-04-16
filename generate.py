from model.model_generator import Model, load_weight
from preprocess.prepare import load_data, to_integer_base, notes_to_midi

if __name__ == "__main__":
    notes = load_data("midi_songs")
    network_input, network_output, vocab_size, pitchnames = to_integer_base(notes, 100)
    model = Model(network_input, vocab_size)
    weight = load_weight("weights.hdf5")
    pattern = model.generate_music(weight, pitchnames=pitchnames)
    notes_to_midi(pattern)
    print("generate")

