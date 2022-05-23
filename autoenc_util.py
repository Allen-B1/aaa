import argparse
from typing import List
import notes.mxl, notes.tensor, notes.midi
from notes.note import Measure, Piece
import autoenc
import pretty_midi
import os.path
import torch
import random
import pandas
import matplotlib.pyplot as plt

MODEL = "saves/autoenc/trial-6/model-50.pt"
SAVE_FOLDER = "saves/autoenc/trial-6"

parser = argparse.ArgumentParser(description="Utilities for autoenc")
parser.add_argument("action", metavar="ACTION", type=str, help="Action to take [gen-file, gen-rand]")
parser.add_argument("--file", type=str, help="File", default=None)
parser.add_argument("--measures", type=int, help="Number of measures to generate", default=32)
parser.add_argument("--epoch", type=int, help="Epoch number of stats file", default=1)
args = parser.parse_args()

model = autoenc.AutoEncoder()
save = torch.load(MODEL, map_location=torch.device('cpu'))
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
        pm.lyrics.append(pretty_midi.Lyric("Epoch " + str(epoch), 0))
        pm.lyrics.append(pretty_midi.Lyric(input_basename, 1/(measures[0].tempo * (1/60))))
        pm.write(SAVE_FOLDER + "/e%d/from/" % epoch + input_basename + ".mid")

elif args.action == "gen-rand":
    code = torch.rand(args.measures, 120)
    measures_tensor = torch.reshape(model.decoder(code), (-1, 49, 88))
    measures = [notes.tensor.from_tensor(measure_tensor) for measure_tensor in measures_tensor]
    encoded_piece = Piece(measures=measures, parts=['piano'])
    pm = notes.midi.to_midi(encoded_piece)
    pm.lyrics.append(pretty_midi.Lyric("Epoch " + str(epoch), 0))
    pm.lyrics.append(pretty_midi.Lyric("Random", 1/(measures[0].tempo * (1/60))))

    id = ''.join([random.choice("abcdefghijklmnopqrstuvwxyx") for i in range(16)])
    print("Genereated " + id + ".mid")
    pm.write(SAVE_FOLDER + "/e%d/rand/" % epoch + id + ".mid")
elif args.action == "loss":
    stats_file = SAVE_FOLDER + "/stats/epoch-" + str(args.epoch) + ".csv"
    df = pandas.read_csv(stats_file)
    losses = df['loss']

    plt.figure()
    plt.title("Epoch " + str(args.epoch))
    plt.plot(losses)
    plt.xlabel("Sample #")
    plt.ylabel("Loss")
    plt.show()
elif args.action == "show-code":
    if args.file is None:
        print("--file must be specified")    
    else:
        piece = notes.mxl.parse_file(args.file)
        codes = []
        for measure in piece.measures:
            code = model.get_code(notes.tensor.to_tensor(measure))
            codes.append([codeitem.item() for codeitem in code])
        df = pandas.DataFrame(codes)
        print(df)

else:
    print("Unknown action: " + args.action)