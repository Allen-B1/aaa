"""Generates what is supposed to be an identity function"""
import sys

from typing import List
sys.path.append(".")

from notes import mxl, midi, tensor
from notes.note import Measure, Part

FILE = "datasets/midide/debussy/deb_prel.musicxml"

parts = mxl.parse_file(FILE)
recreated_measures: List[Measure] = []
for measure in parts[0].measures:
    rec_measure = tensor.from_tensor(tensor.to_tensor(measure))
    recreated_measures.append(rec_measure)

part = Part(recreated_measures, 'Piano')
pm = midi.to_midi([part])
pm.write("tests/output/deb_prel.mid")