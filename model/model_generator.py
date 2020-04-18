import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from tensorflow_core.python.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from tensorflow_core.python.keras.callbacks import TensorBoard
from preprocess.prepare import int_to_note


def load_weight(filename):
    res = tf.keras.models.load_model(filename)
    return res


def create_model(network_input
                 , vocab_size
                 , dropout_value=0.3
                 , units=512
                 , middle_units=256
                 , activation_function='softmax'
                 , loss_function='categorical_crossentropy'
                 , optimizer='rmsprop'):
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
    # set callbacks
    filepath = "training_weight/weights-save-{epoch:04d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    NAME = "prequel_generator_version_1-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    # callbacks
    callbacks_list = [checkpoint]
    return model, callbacks_list


def train(model, network_input, network_output, callbacks_list=[], epochs=200, batch_size=64):
    model.fit(network_input
              , network_output
              , epochs=epochs
              , batch_size=batch_size
              , callbacks=callbacks_list
              )
    weigh = model.get_weights();
    pklfile = "modelweights.pkl"
    fpkl = open(pklfile, 'wb')  # Python 3
    pickle.dump(weigh, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
    fpkl.close()


def generate_music(model, network_input, vocab_size, pitchnames, nb_notes=500, start_notes=None,
                   start_notes_random=True, ):
    if start_notes_random:
        start_notes = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start_notes]
    prediction_output = []
    for note_index in range(nb_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab_size)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note(pitchnames)[index]
        prediction_output.append(result)
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]
    return prediction_output
