from numpy import float128
import torch
import time
import glob
import pandas # type: ignore
import pretty_midi # type: ignore
import numpy as np
from abc import *
import itertools
from torch.utils.data import Dataset

from typing import Any, NamedTuple, Sequence

class Note(NamedTuple):
    pitch: int
    duration: float
    start: float
    step: float

    def pitch_name(self):
        pretty_midi.note_number_to_name(self.pitch)

    def from_midi(notes: Sequence[pretty_midi.Note]) -> 'list[Note]':
        out = []
        notes = list(notes)
        notes.sort(key=lambda note: note.start)

        prev_start: float = 0.0
        for note in notes:
            out.append(Note(pitch=note.pitch, duration=note.end-note.start, start=note.start, step=note.start-prev_start))
            prev_start = note.start
        return out

class NotesRepr(ABC):
    """Represents an encoding of list[Note] <-> Tensor"""
    @abstractmethod
    def encode(self, notes: Sequence[Note]) -> torch.Tensor: pass

    @abstractmethod
    def decode(self, tensor: torch.Tensor) -> Sequence[Note]: pass

class NotesRepr1(NotesRepr):
    """Encodes each note as a tensor [pitch, duration, start, step]"""
    def __init__(self, maxnotes=4096):
        self.maxnotes = maxnotes

    def encode(self, notes: Sequence[Note]) -> torch.Tensor:
        tensor = torch.tensor(notes[:self.maxnotes],dtype=torch.float32)
        if tensor.shape[0] < self.maxnotes:
            zeros = torch.zeros((self.maxnotes, 4), dtype=torch.float32)
            zeros[0:tensor.shape[0],:] = tensor
            return zeros
        else:
            return tensor

    def decode(self, tensor: torch.Tensor) -> Sequence[Note]:
        notes: list[Note] = []
        for row in tensor:
            notes.append(Note(*row))
        return notes

class NotesRepr2(NotesRepr):
    """Fills up frames"""
    def __init__(self, step: float, maxframes: int):
        self.step = step
        self.maxframes = maxframes
    
    def encode(self, notes: Sequence[Note]) -> torch.Tensor:
        # pitches go from [21, 108], which len = 88
        # hopefully 65536 is more than enough frames
        tensor = torch.zeros((self.maxframes, 88), dtype=torch.float64)
        for note in notes:
            frame_index = int(note.start / self.step)
            if frame_index >= self.maxframes: break
            tensor[frame_index][note.pitch - 21] = 1
        return tensor

    def decode(self, tensor: torch.Tensor) -> Sequence[Note]:
        notes: list[Note] = []
        for frame_num, frame in enumerate(tensor):
            start = frame_num*self.step
            for col_num, data in tensor:
                exists = data.item() != 0
                pitch = col_num + 21
                if exists:
                    prev_note = notes[len(notes)-1]
                    notes.append(Note(pitch=pitch, duration=self.step, start=start, step=start - prev_note.start))
        return notes

class MaestroDataset(Dataset):
    _PATH = "maestro-v2.0.0"
    def __init__(self, split: str, repr: NotesRepr):
        self.repr = repr
        self.split = split
        self.labels = pandas.read_csv(MaestroDataset._PATH + "/maestro-v2.0.0.csv")
        self.labels = self.labels[self.labels['split'] == split].reset_index(drop=True)
        self.composer_table = dict(zip(pandas.unique(self.labels['canonical_composer']), itertools.count()))
    
    def __len__(self):
        return len(self.labels)

    def get_midi(self, idx: int) -> pretty_midi.PrettyMIDI:
        data = self.labels.iloc[[idx]]
        midi_file: str = data['midi_filename'][idx]
        midi = pretty_midi.PrettyMIDI(MaestroDataset._PATH + "/" + midi_file)
        return midi

    def __getitem__(self, idx: int):
        data = self.labels.iloc[[idx]]
        midi_file: str = data['midi_filename'][idx]
        midi = pretty_midi.PrettyMIDI(MaestroDataset._PATH + "/" + midi_file)

        composer: str = data['canonical_composer'][idx]
#        tic = time.perf_counter()
        notes = Note.from_midi(midi.instruments[0].notes)
#        toc = time.perf_counter()
#        print("pretty_midi -> list[Note]: %fs" % (toc-tic))
        tensor = self.repr.encode(notes)
#        tic = time.perf_counter()
#        print("list[Note] -> Tensor: %fs" % (tic-toc))
        return tensor, self.composer_table[composer]

class Cache(Dataset):
    def __init__(self, inner: Dataset):
        self.inner = inner
        self.cache: dict[int, Any] = dict()
    
    def __len__(self):
        return self.inner.__len__()
    
    def __getitem__(self, idx: int):
        if idx in self.cache:
            return self.cache[idx]
        self.cache[idx] = self.inner.__getitem__(idx)
        return self.cache[idx]