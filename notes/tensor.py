from typing import Dict, List, Sequence
from .note import Measure, Note
import torch
import math

def to_tensor(measure: Measure) -> torch.Tensor:
    tensor = torch.zeros((49, 88))
    for notelist in measure.notes:
        for position, notes in notelist.items():
            frame_idx = int(round(position * 48 / measure.beats()))
            for note in notes:
                tensor[frame_idx][note.pitch] = note.duration
    tensor[48][0] = measure.time_sig[0]
    tensor[48][1] = measure.time_sig[1]
    tensor[48][2] = measure.tempo
    return tensor

def from_tensor(tensor: torch.Tensor, min_duration=0.125) -> Measure:
    time_sig = (max(2, int(round(tensor[48][0].item()))), int(2**round(math.log2(tensor[48][1].item()))))
    beats = time_sig[0] / time_sig[1] * 4
    tempo = tensor[48][2].item()

    notes: Dict[float, List[Note]] = dict()
    for frame_idx, row in enumerate(tensor[:48]):
        for pitch in row.nonzero():
            if row[pitch].item() < min_duration:
                continue
            notes[frame_idx*beats / 48] = notes.get(frame_idx*beats/48, [])
            notes[frame_idx*beats/48].append(Note(int(pitch.item()), duration=row[pitch].item()))
    return Measure([notes], time_sig, tempo)