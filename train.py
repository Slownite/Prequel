from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
from tensorflow_core.python.keras.utils import np_utils
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from tensorflow_core.python.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization


# In[3]:


def load_data(directory='midi_songs'):
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


def create_model():
    units = 512
    middle_units = 256
    dropout_value = 0.3
    activation_function = 'softmax'
    loss_function = 'categorical_crossentropy'
    optimizer = 'rmsprop'
    model = Sequential()
    model.add(LSTM(units
                   , input_shape=(network_input.shape[1], network_input.shape[2])
                   , recurrent_dropout=dropout_value
                   , return_sequences=True
                   ))
    model.add(LSTM(units, return_sequences=True, recurrent_dropout=dropout_value, ))
    model.add(LSTM(units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_value))
    model.add(Dense(middle_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_value))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_value))
    model.add(Dense(vocab_size))
    model.add(Activation(activation_function))
    model.compile(loss=loss_function
                  , optimizer=optimizer
                  )
    return model


if __name__ == "__main__":
    import time

    filepath = "training_weight/weights-save-{epoch:04d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    NAME = "prequel_generator_version_1-{}".format(int(time.time()))
    # callbacks
    callbacks_list = [checkpoint]
    notes = load_data()
    network_input, network_output, vocab_size, pitchnames = to_integer_base(notes, 100)
    model = create_model()
    epochs = 200
    model.fit(network_input
              , network_output
              , epochs=epochs
              , batch_size=90
              , callbacks=callbacks_list
              )
    model.save('model.h5')
