from typing import Dict, List, Sequence
from .note import Part, Measure, Note
import torch
import math

def to_tensor(measure: Measure) -> torch.Tensor:
    tensor = torch.zeros((49, 88))
    for position, notes in measure.notes.items():
        frame_idx = int(position * 48 / measure.beats)
        for note in notes:
            tensor[frame_idx][note.pitch] = note.duration
    tensor[48][0] = measure.beats
    return tensor

def from_tensor(tensor: torch.Tensor) -> Measure:
    beats = float(round(tensor[48][0].item()))

    notes: Dict[float, List[Note]] = dict()
    for frame_idx, row in enumerate(tensor[:48]):
        for pitch in row.nonzero():
            notes[frame_idx*beats / 48] = notes.get(frame_idx*beats/48, [])
            notes[frame_idx*beats/48].append(Note(int(pitch.item()), duration=row[pitch].item()))
    return Measure(notes, beats=beats) # type: ignore