import requests
from music21 import instrument, chord, note, stream


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
    midi_stream.write('midi', fp="." + filename + '.mid')


if __name__ == "__main__":
    url = "http://localhost:3000/"
    message = requests.post(url)
    print(message.json())
    music = message.json()['message']
    notes_to_midi(music)
