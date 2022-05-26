import torch
import rnn
import autoenc
import argparse
from notes.note import *
import notes.mxl, notes.tensor, notes.midi
import os.path

AUTOENC_MODEL = "saves/autoenc/trial-13/model-2000.pt"
RNN_SAVE_FOLDER = "saves/rnn/trial-2"
RNN_MODEL = RNN_SAVE_FOLDER + "/hi-500.pt"

def get_mxl(args: argparse.Namespace) -> str:
    if 'piece' in vars(args):
        return "datasets/midide/" + args.piece + ".musicxml"
    else:
        return args.file

def gen_file(args: argparse.Namespace): 
    n_measures: int = args.measures

    autoenc_model, autoenc_epochs = autoenc.load(AUTOENC_MODEL, "cpu")
    rnn_model, rnn_epochs, autoenc_version = rnn.MeasurePredictor.load(RNN_MODEL, "cpu")
    assert autoenc_model.version() == autoenc_version, "RNN and AutoEncoder have different autoenc_version (%d vs %d)" % (autoenc_version, autoenc_model.version())

    mxl = get_mxl(args)
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
    pm = notes.midi.to_midi(piece)
    pm.write(RNN_SAVE_FOLDER + "/" + os.path.splitext(os.path.basename(mxl))[0] + ".mid")

parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers(help="possible commands", required=True)

gen_file_subcommand = subcommands.add_parser("gen-file", help="generate-file")
gen_file_subcommand.add_argument("--file", type=str)
gen_file_subcommand.add_argument("--piece", type=str)
gen_file_subcommand.add_argument("--measures", type=int, default=16)
gen_file_subcommand.set_defaults(func=gen_file)

args = parser.parse_args()
args.func(args)