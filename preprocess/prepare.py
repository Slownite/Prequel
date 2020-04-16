from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
from tensorflow_core.python.keras.utils import np_utils


def load_data(directory):
    notes = []
    for file in glob.glob(directory + "/*.mid"):
        midi_file = converter.parse(file)
        parts = instrument.partitionByInstrument(midi_file)
        if parts:
            decomposition_file = parts.parts[0].recurse()
        else:
            decomposition_file = midi_file.flat.notes

        for element in decomposition_file:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def to_integer_base(notes, sequence_size):
    vocab_size = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    notes_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_output = []
    network_input = []
    for i in range(0, len(notes) - sequence_size, 1):
        sequence_in = notes[i:i + sequence_size]
        sequence_out = notes[i + sequence_size]
        network_input.append([notes_to_int[char] for char in sequence_in])
        network_output.append(notes_to_int[sequence_out])

    patterns = len(network_input)
    network_input = np.reshape(network_input, (patterns, sequence_size, 1))
    network_input = network_input / float(vocab_size)
    network_output = np_utils.to_categorical(network_output)
    return network_input, network_output, vocab_size, pitchnames


def int_to_note(pitchnames):
    res = dict((number, note) for number, note in enumerate(pitchnames))
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
    midi_stream.write('midi', fp=filename+'.mid')
