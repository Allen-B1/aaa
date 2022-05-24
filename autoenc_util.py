import argparse
from posixpath import basename
from typing import List, Union
import notes.mxl, notes.tensor, notes.midi
from notes.note import Measure, Piece
import autoenc
import pretty_midi
import os.path
import torch
import random
import pandas
import matplotlib.pyplot as plt

MODEL = autoenc.SAVE_FOLDER + "/model-5.pt"
SAVE_FOLDER = autoenc.SAVE_FOLDER
model, epoch = autoenc.load(MODEL, "cpu")

parser = argparse.ArgumentParser(description="Utilities for autoenc")
parser.add_argument("action", metavar="ACTION", type=str, help="Action to take [gen-file, gen-rand]")
parser.add_argument("--file", type=str, help="File", default=None)
parser.add_argument("--piece", type=str, help="Piece from the midide dataset", default=None)
parser.add_argument("--measures", type=int, help="Number of measures to generate", default=32)
parser.add_argument("--epoch", type=int, help="Epoch number of stats file", default=1)
parser.add_argument("--epoch-to", type=int, help="Epoch number of stats file", default=None)
parser.add_argument("--measure-num", type=int, default=1)
args = parser.parse_args()

try:
    os.makedirs(SAVE_FOLDER + "/e%d/rand" % epoch)
except FileExistsError: pass
try:
    os.makedirs(SAVE_FOLDER + "/e%d/from" % epoch)
except FileExistsError: pass
try:
    os.makedirs(SAVE_FOLDER + "/e%d/codes" % epoch)
except FileExistsError: pass

mxl_file: Union[str, None] = args.file if args.file is not None else ("datasets/midide/" + args.piece + ".musicxml" if args.piece is not None else None)

if args.action == "gen-file":
    if mxl_file is None:
        print("--file or --piece must be specified")    
    else:
        piece = notes.mxl.parse_file(mxl_file)
        measures: List[Measure] = []
        for measure in piece.measures:
            encoded_measure = notes.tensor.from_tensor(torch.reshape(model(notes.tensor.to_tensor(measure)), shape=(49, 88)), min_duration=1/8)
            measures.append(encoded_measure)
        encoded_piece = Piece(measures, parts=['piano'])
        pm = notes.midi.to_midi(encoded_piece)

        input_basename = os.path.splitext(os.path.basename(mxl_file))[0]
        pm.lyrics.append(pretty_midi.Lyric("Epoch " + str(epoch), 0))
        pm.lyrics.append(pretty_midi.Lyric(input_basename, 1/(measures[0].tempo * (1/60))))
        pm.write(SAVE_FOLDER + "/e%d/from/" % epoch + input_basename + ".mid")

elif args.action == "gen-rand":
    code = torch.rand(args.measures, 120)
    measures_tensor = torch.reshape(model.decode(code), (-1, 49, 88))
    measures = [notes.tensor.from_tensor(measure_tensor) for measure_tensor in measures_tensor]
    encoded_piece = Piece(measures=measures, parts=['piano'])
    pm = notes.midi.to_midi(encoded_piece)
    pm.lyrics.append(pretty_midi.Lyric("Epoch " + str(epoch), 0))
    pm.lyrics.append(pretty_midi.Lyric("Random", 1/(measures[0].tempo * (1/60))))

    id = ''.join([random.choice("abcdefghijklmnopqrstuvwxyx") for i in range(16)])
    print("Genereated " + id + ".mid")
    pm.write(SAVE_FOLDER + "/e%d/rand/" % epoch + id + ".mid")
elif args.action == "loss":
    if args.epoch_to == None:
        stats_file = SAVE_FOLDER + "/stats/epoch-" + str(args.epoch) + ".csv"
        df = pandas.read_csv(stats_file)
        losses = df['loss']

        plt.figure()
        plt.title("Epoch " + str(args.epoch))
        plt.plot(losses)
        plt.xlabel("Sample #")
        plt.ylabel("Loss")
        plt.show()
    else:
        from_: int = args.epoch
        to: int = args.epoch_to

        df = pandas.read_csv(SAVE_FOLDER + "/stats/epochs-%d-to-%d.csv" % (from_, to))
        losses = df['loss']
        epochs = df['epoch']

        plt.figure()
        plt.title("Epochs %d to %d" % (from_, to))
        plt.plot(epochs, losses)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()

elif args.action == "show-code":
    if mxl_file is None:
        print("--file or --piece must be specified")    
    else:
        piece = notes.mxl.parse_file(mxl_file)
        codes: List[List[float]] = []
        for measure in piece.measures:
            code = model.encode(notes.tensor.to_tensor(measure))
            codes.append([codeitem.item() for codeitem in code])
        df = pandas.DataFrame(codes)
        filepath = SAVE_FOLDER + "/e" + str(epoch) + "/codes/" + basename(mxl_file) + ".csv"
        df.to_csv(filepath)
        print("Saved to: " + filepath)
        unused = set(range(0, 120))
        for code_ in codes:
            unused_tmp = set(idx for idx, codeitem in enumerate(code_) if codeitem == 0)
            unused = unused.intersection(unused_tmp)
        print("unused: indexes", unused)
        print("unused: %d/%d"% (len(unused), 120))

elif args.action == "show-output":
    if mxl_file is None:
        print("--file or --piece must be specified")    
    else:
        piece = notes.mxl.parse_file(mxl_file)
        measure = piece.measures[args.measure_num - 1]
        measure_tensor = notes.tensor.to_tensor(measure)
        output_tensor = torch.reshape(model(measure_tensor), (49, 88))
        torch.set_printoptions(threshold=10000)
        print(measure_tensor)
        print(output_tensor)
else:
    print("Unknown action: " + args.action)