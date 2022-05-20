from typing import NamedTuple, Mapping, Sequence, Tuple

class Note(NamedTuple):
    pitch: int       # number; MIDI pitch - 21; [0, 88)
    duration: float  # quarter note = 1

    def pitch_name(self) -> str:
        import pretty_midi  # type: ignore
        return pretty_midi.note_number_to_name(self.pitch + 21)

    def __repr__(self) -> str:
        assert type(self.duration) == float
        return "Note(\"%s\", pitch=%s, duration=%f)" % (self.pitch_name(), repr(self.pitch), self.duration)

NoteList = Mapping[float, Sequence[Note]]

class Measure(NamedTuple):
    notes: Sequence[NoteList]
    time_sig: Tuple[float, float] # 6/8
    tempo: float # bpm

    def beats(self) -> float:
        return self.time_sig[0] / self.time_sig[1] * 4


class Piece(NamedTuple):
    measures: Sequence[Measure]
    parts: Sequence[str]
