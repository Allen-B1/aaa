import random
import pretty_midi
import mido

from typing import Sequence, List
from .note import Piece

def to_midi(piece: Piece) -> pretty_midi.PrettyMIDI:
    # (beats/min * min/sec)^-1

    tempo = piece.measures[0].tempo
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for part in piece.parts:
        program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano' if part.lower() == 'piano' else part)
        instrument = pretty_midi.Instrument(program=program)
        midi.instruments.append(instrument)

    current_time = 0.0
    for measure in piece.measures:
        tempo = measure.tempo
        
        sec_per_beats = 1/(tempo * (1/60))
        midi.time_signature_changes.append(pretty_midi.TimeSignature(measure.time_sig[0], measure.time_sig[1], current_time))
        midi._tick_scales.append((midi.time_to_tick(current_time), 60.0/(tempo*midi.resolution)))
        midi._update_tick_to_time(midi.time_to_tick(current_time+5000))
        for part_id, part_notes in enumerate(measure.notes):
            for position, notes in part_notes.items():
                note_time = current_time + position*sec_per_beats
                for note in notes:
                    midi.instruments[part_id].notes.append(pretty_midi.Note(velocity=96,pitch=note.pitch+21,start=note_time,end=note_time + note.duration*sec_per_beats))

        current_time += measure.beats() * sec_per_beats

    return midi