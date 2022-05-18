from typing import Dict, List, Sequence
from .note import Part, Measure, Note
import torch
import math

def to_tensor(measure: Measure) -> torch.Tensor:
    tensor = torch.zeros((48, 88))
    for position, notes in measure.notes.items():
        frame_idx = int(position * 48 / 4)
        for note in notes:
            tensor[frame_idx][note.pitch] = note.duration
    return tensor

def from_tensor(tensor: torch.Tensor) -> Measure:
    notes: Dict[float, List[Note]] = dict()
    for frame_idx, row in enumerate(tensor):
        for pitch in row.nonzero():
            notes[frame_idx*4 / 48] = notes.get(frame_idx*4/48, [])
            notes[frame_idx*4/48].append(Note(pitch, row[pitch]))
    return Measure(notes, beats=math.nan) # type: ignore