from typing import *

class Note(NamedTuple):
    pitch: int      # number
    duration: float # quarter note = 1
                    # can be negative if rest
    def pitch_name(self) -> str:
        import pretty_midi # type: ignore
        return pretty_midi.note_number_to_name(self.pitch)

    def __repr__(self) -> str:
        return "Note(\"%s\", pitch=%s, duration=%f)" % (self.pitch_name(), repr(self.pitch), self.duration) 

class Measure(NamedTuple):
    notes: Dict[float, Sequence[Note]]
    beats: float

class Part(NamedTuple):
    measures: Sequence[Measure]
    instrument: str
