from typing import *
import pretty_midi # type: ignore
from abc import *


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
