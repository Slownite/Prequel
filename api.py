from abc import ABC

from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop

from generate import prepare_sequences, load_weight, generate
from train import load_data


class MusicGenerator(RequestHandler, ABC):
    def get(self):
        self.write({'message': 'hello world'})

    def post(self):
        print("preparing")
        message = generate(model, network_input, pitchnames, n_vocab)
        self.write({'message': message})


def make_app():
    urls = [("/", MusicGenerator)]
    return Application(urls)


if __name__ == '__main__':
    notes = load_data()
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = load_weight()
    app = make_app()
    port = 3000
    app.listen(port)
    print("Listen on port:", port)
    IOLoop.instance().start()
