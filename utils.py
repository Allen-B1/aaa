import argparse
import random
from typing import Union
from notes.note import Piece
import notes.midi
import os.path
import pretty_midi

def get_musicxml(args: argparse.Namespace) -> Union[str, None]:
    if args.midide is not None:
        return "datasets/midide/" + args.midide + ".musicxml"
    elif args.file is not None:
        return args.file
    else:
        return None

def add_musicxml_options(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--midide", help="MusicXML from the midide dataset to load", type=str)
    group.add_argument("--file", help="File to load", type=str)

def save_piece(piece: Piece, folder: str, filename: Union[str, None], epoch: int):
    """Saves .mid and .msk2 files in a standardized manner"""
    pm = notes.midi.to_midi(piece)
    pm.lyrics.append(pretty_midi.Lyric("Epoch: " + str(epoch), 0))
    if filename is not None:
        basename = os.path.splitext(os.path.basename(filename))[0]
        pm.lyrics.append(pretty_midi.Lyric("From: " + basename, 1/(piece.measures[0].tempo * (1/60))))
    else:
        basename = "random-" + "".join([random.choice("abcdefghijklmnopqrstuvwxyx") for i in range(8)])
        pm.lyrics.append(pretty_midi.Lyric("Random", 1/(piece.measures[0].tempo * (1/60))))
    pm.write(folder + "/%s-e%d.mid" % (basename, epoch))

    import pickle
    with open(folder + "/%s-e%d.msk2" % (basename, epoch), "wb") as f:
        pickle.dump({
            "epoch": epoch,
            "random": filename is None,
            "piece": piece,
        }, f)

if __name__ == "__main__":
    import pickle
    import notes.mxl, notes.midi
    import os.path
    def convert_from(args: argparse.Namespace):
        mxl_path = get_musicxml(args)
        assert mxl_path is not None
        piece = notes.mxl.parse_file(mxl_path)
        with open(args.out if args.out is not None else ("out/" + os.path.splitext(os.path.basename(mxl_path))[0] + ".msk"), "wb") as f:
            pickle.dump(piece, f)

    def convert_to(args: argparse.Namespace): 
        with open(args.file, "rb") as f:
            piece = pickle.load(f)

        out_file: str = args.out
        if out_file.endswith(".mid") or out_file.endswith(".midi"):
            pm = notes.midi.to_midi(piece)
            pm.write(args.out)
        else:
            print("Unknown file format")
    

    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(title="actions")
    
    from_parser = subcommands.add_parser("from")
    add_musicxml_options(from_parser)
    from_parser.add_argument("--out", help="Output MSK file", type=str, default=None)
    from_parser.set_defaults(func=convert_from)

    to_parser = subcommands.add_parser("to")
    to_parser.add_argument("--file", help="MSK file to convert from", required=True)
    to_parser.add_argument("--out", help="Output file", type=str, required=True)
    to_parser.set_defaults(func=convert_to)

    args = parser.parse_args()
    args.func(args)