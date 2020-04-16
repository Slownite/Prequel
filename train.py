import sys

from model.model_generator import create_model, train, generate_music
from preprocess.prepare import load_data, to_integer_base, notes_to_midi

if __name__ == "__main__":
    args = sys.argv
    notes = load_data("midi_songs")
    network_input, network_output, vocab_size, pitchnames = to_integer_base(notes, 100)
    model, callbacks = create_model(network_input, vocab_size, units=int(args[2]), middle_units=args[3])
    train(model, network_input, network_output, callbacks_list=callbacks, epochs=int(args[1]))
    output = generate_music(model, network_input, vocab_size, pitchnames)
    notes_to_midi(output)
