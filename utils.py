import argparse
from typing import Union

def get_musicxml(args: argparse.Namespace) -> Union[str, None]:
    if 'midide' in vars(args):
        return "datasets/midide/" + args.midide + ".musicxml"
    elif 'file' in vars(args):
        return args.file
    else:
        return None

def add_musicxml(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--midide", help="MusicXML from the midide dataset to load", type=str)
    group.add_argument("--file", help="File to load", type=str)
