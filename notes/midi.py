import pretty_midi

from typing import Sequence
from .note import Part

def to_midi(parts: Sequence[Part], tempo: int=120) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    # (beats/min * min/sec)^-1
    sec_per_beats = 1/(tempo * (1/60))
    for part in parts:
        program = pretty_midi.instrument_name_to_program('piano')
        instrument = pretty_midi.Instrument(program=program)
        current_time = 0.0
        for measure in part.measures:
            midi.time_signature_changes.append(pretty_midi.TimeSignature(measure.beats, 4, current_time))
            for position, notes in measure.notes.items():
                position_time = current_time + position*sec_per_beats
                for note in notes:
                    instrument.notes.append(pretty_midi.Note(velocity=96,pitch=note.pitch+21,start=position_time,end=position_time + note.duration*sec_per_beats))

            current_time += measure.beats * sec_per_beats
        midi.instruments.append(instrument)
    return midi