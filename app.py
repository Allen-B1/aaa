import flask
import notes.abc, notes.tensor, notes.midi
from notes.note import *
import rnn
import convrnn
import pretty_midi
import autoenc
import voices
import random
import os

MODELS = {
    "rnn": "./saves/rnn/trial-5/model-1000.pt",
    "convrnn": "./saves/convrnn/trial-4/model-200.pt",
    "autoenc": "./saves/autoenc/trial-15/model-2000.pt"
}

app = flask.Flask(__name__, template_folder='site', static_folder='site/static')

def create_waveform(pm: pretty_midi.PrettyMIDI) -> str:
    """Create a waveform, save to ./site/static, return the basename"""
    import numpy as np
    import struct
    import wave
    import random

    signal = pm.fluidsynth(16000)
    signal = (signal * (2**14) / np.max(signal))
    signal = signal.astype('int32')
    data = struct.pack('<' + ('h'*len(signal)), *signal)

    id = "output-" + str.join("", [random.choice("qwertyuuiopasdfghjjklzxcvnbm") for i in range(8)]) + ".wav"
    file = wave.open("site/static/" + id, 'wb')
    file.setnchannels(1)
    file.setsampwidth(2)
    file.setframerate(16000)
    file.writeframes(data)
    file.close()
    return id

@app.route("/")
def form():
    return flask.render_template('form.html')

@app.route("/process1", methods=["POST"])
def process1():
    abc_text = flask.request.form["abc"]
    n_measures = int(flask.request.form["measures"])
    piece = notes.abc.parse_piece(abc_text)
    measure_tensor = notes.tensor.to_tensor(piece.measures[0])
    if flask.request.form["model"] == "convrnn":
        model, _ = convrnn.load(MODELS["convrnn"], "cpu")
        measures = [notes.tensor.from_tensor(measure_tensor)]
        next_tensor = measure_tensor
        hidden = None
        for i in range(n_measures-1):
            next_tensor, hidden = model.predict(next_tensor, hidden)
            measures.append(notes.tensor.from_tensor(next_tensor))
    else: 
        model, _, _ = rnn.MeasurePredictor.load(MODELS["rnn"], "cpu")
        autoenc_model, _ = autoenc.load(MODELS["autoenc"], "cpu")
        code_tensor = autoenc_model.encode(measure_tensor.unsqueeze(0))
        measures = [notes.tensor.from_tensor(autoenc_model.decode_regularize(code_tensor).squeeze())]
        hidden = None
        for i in range(n_measures-1):
            code_tensor, hidden = model.predict(code_tensor, hidden)
            code_tensor = code_tensor[0]
            decoded = autoenc_model.decode_regularize(code_tensor).squeeze()
            print(decoded.shape)
            measures.append(notes.tensor.from_tensor(decoded))
    piece = Piece(measures=measures, parts=["Piano"])
    piece_orchestrated = voices.orchestrate(piece)
    pm = notes.midi.to_midi(piece)
    pm_orchestrated = notes.midi.to_midi(piece_orchestrated)

    id = str.join("", [random.choice("qwertyuuiopasdfghjjklzxcvnbm") for i in range(8)])
    output_dir = "site/static/output/" + id
    os.makedirs(output_dir)
    pm.write(output_dir + "/piano.mid")
    pm_orchestrated.write(output_dir + "/orch.mid")

    return flask.render_template("processing.html", id=id)

@app.route("/process2/<id>", methods=["GET"])
def process2(id: str):
    output_dir = "site/static/output/" + id
    os.system("flatpak run org.musescore.MuseScore -o '%s' '%s'" % (output_dir + "/piano.pdf", output_dir + "/piano.mid"))
    os.system("flatpak run org.musescore.MuseScore -o '%s' '%s'" % (output_dir + "/orch.pdf",  output_dir + "/orch.mid"))
    os.system("flatpak run org.musescore.MuseScore -o '%s' '%s'" % (output_dir + "/piano.ogg",  output_dir + "/piano.mid"))
    os.system("flatpak run org.musescore.MuseScore -o '%s' '%s'" % (output_dir + "/orch.ogg",  output_dir + "/orch.mid"))

    return flask.redirect("/output/" + id)

@app.route("/output/<id>")
def output(id: str):
    return flask.render_template("output.html", dir="/static/output/" + id)