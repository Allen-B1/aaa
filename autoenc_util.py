import argparse
from typing import List
import notes.mxl, notes.tensor, notes.midi
from notes.note import Measure, Piece
import autoenc
import pretty_midi
import os.path
import torch
import random

MODEL = "saves/autoenc/trial-4/model1.pt"
SAVE_FOLDER = "saves/autoenc/trial-4"

parser = argparse.ArgumentParser(description="Utilities for autoenc")
parser.add_argument("action", metavar="ACTION", type=str, help="Action to take [gen-file, gen-rand]")
parser.add_argument("--file", type=str, help="file", default=None)
args = parser.parse_args()

model = autoenc.AutoEncoder()
save = torch.load(MODEL)
model.load_state_dict(save["model"])
epoch: int = save["epoch"]

try:
    os.makedirs(SAVE_FOLDER + "/e%d/rand" % epoch)
except FileExistsError: pass
try:
    os.makedirs(SAVE_FOLDER + "/e%d/from" % epoch)
except FileExistsError: pass

if args.action == "gen-file":
    if args.file is None:
        print("--file must be specified")    
    else:
        piece = notes.mxl.parse_file(args.file)
        measures: List[Measure] = []
        for measure in piece.measures:
            encoded_measure = notes.tensor.from_tensor(torch.reshape(model(notes.tensor.to_tensor(measure)), shape=(49, 88)), min_duration=1/8)
            measures.append(encoded_measure)
        encoded_piece = Piece(measures, parts=['piano'])
        pm = notes.midi.to_midi(encoded_piece)

        input_basename = os.path.splitext(os.path.basename(args.file))[0]
        pm.lyrics.append(pretty_midi.Lyric(input_basename, 0))
        pm.write(SAVE_FOLDER + "/e%d/from/" % epoch + input_basename + ".mid")

elif args.action == "gen-rand":
    code = torch.rand(120)
    measure_tensor = torch.reshape(model.decoder(code), (49, 88))
    measure = notes.tensor.from_tensor(measure_tensor)
    encoded_piece = Piece(measures=[measure], parts=['piano'])
    pm = notes.midi.to_midi(encoded_piece)

    id = ''.join([random.choice("abcdefghijklmnopqrstuvwxyx") for i in range(16)])
    print("Genereated " + id + ".mid")
    pm.write(SAVE_FOLDER + "/e%d/rand/" % epoch + id + ".mid")
else:
    print("Unknown action: " + args.action)