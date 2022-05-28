import argparse
from typing import Union

def get_musicxml(args: argparse.Namespace) -> Union[str, None]:
    if 'midide' in vars(args):
        return "datasets/midide/" + args.midide + ".musicxml"
    elif 'file' in vars(args):
        return args.file
    else:
        return None

def add_musicxml_options(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--midide", help="MusicXML from the midide dataset to load", type=str)
    group.add_argument("--file", help="File to load", type=str)

if __name__ == "__main__":
    import pickle
    import notes.mxl, notes.midi
    import os.path
    def convert_from(args: argparse.Namespace):
        mxl_path = get_musicxml(args)
        assert mxl_path is not None
        piece = notes.mxl.parse_file(mxl_path)
        with open(args.out if args.out is not None else ("output/" + os.path.splitext(os.path.basename(mxl_path))[0] + ".msk"), "wb") as f:
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
    from_parser.add_argument("--out", help="Output file", type=str, required=True)
    to_parser.set_defaults(func=convert_to)

    args = parser.parse_args()
    args.func(args)