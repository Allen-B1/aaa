import torch
import predictor
import autoenc
import argparse
from notes.note import *
import notes.mxl, notes.tensor, notes.midi
import os.path
import utils

AUTOENC_MODEL = "saves/autoenc/trial-15/model-2000.pt"
PREDICTOR_SAVE_FOLDER = "saves/predictor/trial-5"
PREDICTOR_MODEL = PREDICTOR_SAVE_FOLDER + "/model-1000.pt"


def gen_file(args: argparse.Namespace): 
    n_measures: int = args.measures

    autoenc_model, autoenc_epochs = autoenc.load(AUTOENC_MODEL, "cpu")
    predictor_model, predictor_epochs, autoenc_version = predictor.MeasurePredictor.load(PREDICTOR_MODEL, "cpu")
    assert autoenc_model.version() == autoenc_version, "Predictor and AutoEncoder have different autoenc_version (%d vs %d)" % (autoenc_version, autoenc_model.version())

    mxl = utils.get_musicxml(args)
    assert mxl is not None
    piece = notes.mxl.parse_file(mxl)
    tensor = notes.tensor.to_tensor(piece.measures[0])
    code_tensor = autoenc_model.encode(tensor.unsqueeze(0))[0]

    code_tensors = [code_tensor.clone().detach()]
    for i in range(n_measures):
        print(code_tensor)
        code_tensor = predictor_model(code_tensor.unsqueeze(0))
        code_tensor = code_tensor[0]
        code_tensors.append(code_tensor.clone().detach())

    measures_tensor = autoenc_model.decode_regularize(torch.stack(code_tensors))
    measures = [notes.tensor.from_tensor(measure[0]) for measure in measures_tensor]
    piece = Piece(measures, parts=["piano"])
    utils.save_piece(piece, PREDICTOR_SAVE_FOLDER, mxl, predictor_epochs)

parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers(metavar="COMMANDS", required=True)

gen_file_subcommand = subcommands.add_parser("gen-file", help="generate-file")
utils.add_musicxml_options(gen_file_subcommand)
gen_file_subcommand.add_argument("--measures", type=int, default=16)
gen_file_subcommand.set_defaults(func=gen_file)

args = parser.parse_args()
args.func(args)
