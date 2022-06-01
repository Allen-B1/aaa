import random
import torch
import utils
import notes.mxl, notes.tensor, notes.midi
from notes.note import *
import argparse
import os.path, os

SAVE_FOLDER = "saves/convrnn/trial-4"

import convrnn
model, epoch = convrnn.load(SAVE_FOLDER + "/model-200.pt", "cpu")

def gen_file(args: argparse.Namespace):
    mxl_file = utils.get_musicxml(args)
    assert mxl_file is not None
    piece = notes.mxl.parse_file(mxl_file)
    measure = notes.tensor.to_tensor(piece.measures[0])
    measures = [notes.tensor.from_tensor(measure)]
    hidden = None
    for i in range(args.measures):
        measure, hidden = model.predict(measure, hidden)
        measures.append(notes.tensor.from_tensor(measure))

    piece = Piece(measures, parts=['piano'])

    utils.save_piece(piece, SAVE_FOLDER, mxl_file, epoch)

def gen_rand(args: argparse.Namespace):
    measure = torch.rand((49, 88)).apply_(lambda x: 0.25 if random.random() > x else 0)
    measure[48][2] = random.randint(60, 240)
    measures = [notes.tensor.from_tensor(measure)]
    hidden = None
    for i in range(args.measures):
        measure, hidden = model.predict(measure, hidden)
        hidden[0].add_(torch.randn(hidden[0].shape))
        measures.append(notes.tensor.from_tensor(measure))

    piece = Piece(measures, parts=['piano'])
    utils.save_piece(piece, SAVE_FOLDER, None, epoch)


parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers(metavar="ACTION", required=True)

gen_file_parser = subcommands.add_parser("gen-file", help="generate based on first measure of file")
utils.add_musicxml_options(gen_file_parser)
gen_file_parser.add_argument("--measures", type=int, help="Number of measures to generate", default=16)
gen_file_parser.set_defaults(func=gen_file)

gen_rand_parser = subcommands.add_parser("gen-rand", help="generated random" )
gen_rand_parser.add_argument("--measures", type=int, help="Number of measures to generate", default=16)
gen_rand_parser.set_defaults(func=gen_rand)

args = parser.parse_args()
args.func(args)
