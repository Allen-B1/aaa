import argparse
import glob
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
import utils

MODEL = autoenc.SAVE_FOLDER + "/model-2000.pt"
SAVE_FOLDER = autoenc.SAVE_FOLDER

model, epoch = autoenc.load(MODEL, "cpu")
model.eval()

try:
    os.makedirs(SAVE_FOLDER + "/e%d/codes" % epoch)
except FileExistsError: pass

def gen_file(args: argparse.Namespace):
    mxl_file = utils.get_musicxml(args)
    assert mxl_file is not None
    piece = notes.mxl.parse_file(mxl_file)
    measures: List[Measure] = []
    for measure in piece.measures:
        encoded_measure = notes.tensor.from_tensor(torch.reshape(model.decode_regularize(model.encode(notes.tensor.to_tensor(measure).unsqueeze(0))), shape=(49, 88)), min_duration=1/8)
        measures.append(encoded_measure)
    encoded_piece = Piece(measures, parts=['piano'])

    utils.save_piece(encoded_piece, SAVE_FOLDER, mxl_file, epoch)

def gen_rand(args: argparse.Namespace):
    code = torch.rand(args.measures, 120)
    code = torch.sigmoid(code)
    measures_tensor = torch.reshape(model.decode_regularize(code), (-1, 49, 88))
    measures = [notes.tensor.from_tensor(measure_tensor) for measure_tensor in measures_tensor]
    encoded_piece = Piece(measures=measures, parts=['piano'])
    utils.save_piece(encoded_piece, SAVE_FOLDER, None, epoch)

def show_code(args: argparse.Namespace):
    mxl_file = utils.get_musicxml(args)
    assert mxl_file is not None
    piece = notes.mxl.parse_file(mxl_file)
    codes: List[List[float]] = []
    for measure in piece.measures:
        code = model.encode(notes.tensor.to_tensor(measure).unsqueeze(0))[0]
        codes.append([codeitem.item() for codeitem in code])
    df = pandas.DataFrame(codes)
    filepath = SAVE_FOLDER + "/e" + str(epoch) + "/codes/" + basename(mxl_file) + ".csv"
    df.to_csv(filepath)
    print("Saved to: " + filepath)
    unused = set(range(0, 120))
    for code_ in codes:
        unused_tmp = set(idx for idx, codeitem in enumerate(code_) if codeitem == 0)
        unused = unused.intersection(unused_tmp)
    print("unused indexes: ", unused)
    print("unused: %d/%d"% (len(unused), 120))

parser = argparse.ArgumentParser(description="Utilities for autoenc")
subcommands = parser.add_subparsers(metavar="ACTION", required=True)

gen_file_parser = subcommands.add_parser("gen-file", help="encode & decode from file")
utils.add_musicxml_options(gen_file_parser)
gen_file_parser.add_argument("--measures", type=int, help="Number of measures to generate", default=8)
gen_file_parser.set_defaults(func=gen_file)

gen_rand_parser = subcommands.add_parser("gen-rand", help="generate sigmoid-distributed random")
gen_rand_parser.add_argument("--measures", type=int, help="Number of measures to generate", default=8)
gen_rand_parser.set_defaults(func=gen_rand)

show_code_parser = subcommands.add_parser('show-code', help='show code of a piece')
utils.add_musicxml_options(show_code_parser)
show_code_parser.set_defaults(func=show_code)

args = parser.parse_args()
args.func(args)