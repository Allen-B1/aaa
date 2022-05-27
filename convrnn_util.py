import utils
import notes.mxl, notes.tensor, notes.midi
from notes.note import *
import argparse
import os.path

SAVE_FOLDER = "saves/convrnn/trial-3"

parser = argparse.ArgumentParser()
utils.add_musicxml(parser)
parser.add_argument("--measures", type=int, help="Number of measures to generate", default=16)
args = parser.parse_args()

mxl_file = utils.get_musicxml(args)
assert mxl_file is not None
piece = notes.mxl.parse_file(mxl_file)

import convrnn
model, epoch = convrnn.load("saves/convrnn/trial-3/model-200.pt", "cpu")

measure = notes.tensor.to_tensor(piece.measures[0])
measures = [notes.tensor.from_tensor(measure)]
hidden = None
for i in range(args.measures):
    measure, hidden = model.predict(measure, hidden)
    measures.append(notes.tensor.from_tensor(measure))

piece = Piece(measures, parts=['piano'])
pm = notes.midi.to_midi(piece)
pm.write(SAVE_FOLDER + "/e%d/" % epoch + os.path.splitext(os.path.basename(mxl_file))[0] + ".mid")
