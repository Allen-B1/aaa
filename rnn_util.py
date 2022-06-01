import torch
import rnn
import autoenc
import argparse
from notes.note import *
import notes.mxl, notes.tensor, notes.midi
import os.path
import pretty_midi
import utils

AUTOENC_MODEL = "saves/autoenc/trial-15/model-2000.pt"
RNN_SAVE_FOLDER = "saves/rnn/trial-5"
RNN_MODEL = RNN_SAVE_FOLDER + "/model-1000.pt"

autoenc_model, autoenc_epochs = autoenc.load(AUTOENC_MODEL, "cpu")
rnn_model, rnn_epochs, autoenc_version = rnn.MeasurePredictor.load(RNN_MODEL, "cpu")
assert autoenc_model.version() == autoenc_version, "RNN and AutoEncoder have different autoenc_version (%d vs %d)" % (autoenc_version, autoenc_model.version())

def gen_file(args: argparse.Namespace): 
    n_measures: int = args.measures

    mxl = utils.get_musicxml(args)
    assert mxl is not None
    piece = notes.mxl.parse_file(mxl)
    tensor = notes.tensor.to_tensor(piece.measures[0])
    code_tensor = autoenc_model.encode(tensor.unsqueeze(0))[0]

    hidden = None
    code_tensors = [code_tensor.clone().detach()]
    for i in range(n_measures):
        print(code_tensor)
        code_tensor, hidden = rnn_model.predict(code_tensor.unsqueeze(0), hidden)
        code_tensor = code_tensor[0][0]
        code_tensors.append(code_tensor.clone().detach())

    measures_tensor = autoenc_model.decode_regularize(torch.stack(code_tensors))
    measures = [notes.tensor.from_tensor(measure[0]) for measure in measures_tensor]
    piece = Piece(measures, parts=["piano"])

    utils.save_piece(piece, RNN_SAVE_FOLDER, mxl, rnn_epochs)

def gen_rand(args: argparse.Namespace):
    n_measures: int = args.measures

    code_tensor = torch.sigmoid(torch.rand(120))

    hidden = None
    code_tensors = [code_tensor.clone().detach()]
    for i in range(n_measures):
        print(code_tensor)
        code_tensor, hidden = rnn_model.predict(code_tensor.unsqueeze(0), hidden)
        code_tensor = code_tensor[0][0]
        code_tensors.append(code_tensor.clone().detach())

    measures_tensor = autoenc_model.decode_regularize(torch.stack(code_tensors))
    measures = [notes.tensor.from_tensor(measure[0]) for measure in measures_tensor]
    piece = Piece(measures, parts=["piano"])

    utils.save_piece(piece, RNN_SAVE_FOLDER, None, rnn_epochs)

parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers(metavar="ACTION", required=True)

gen_file_subcommand = subcommands.add_parser("gen-file", help="generate from first measure of file")
utils.add_musicxml_options(gen_file_subcommand)
gen_file_subcommand.add_argument("--measures", type=int, default=16)
gen_file_subcommand.set_defaults(func=gen_file)

gen_rand_subcommand = subcommands.add_parser("gen-rand", help="generate random")
gen_rand_subcommand.add_argument("--measures", type=int, default=16)
gen_rand_subcommand.set_defaults(func=gen_rand)

args = parser.parse_args()
args.func(args)
