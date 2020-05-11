import numpy as np
from music21 import *
import tensorflow as tf
from train import load_data


def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def prepare_sequences(notes, pitchnames, n_vocab):

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input


def load_weight(filename='model.h5'):
    res = tf.keras.models.load_model(filename)
    return res


def notes_to_midi(output, filename='output', offset=0):
    output_notes = []
    for pattern in output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp="../output/" + filename + '.mid')


def generate(model, network_input, pitchnames, n_vocab):
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    return prediction_output


if __name__ == "__main__":

    notes = load_data("../midi_songs")
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = load_weight()
    prd = generate(model, network_input, pitchnames, n_vocab)
    notes_to_midi(prd)

