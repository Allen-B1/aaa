from collections import defaultdict
from typing import DefaultDict, Dict, List, MutableSet

from notes.note import *

def orchestrate(piece: Piece) -> Piece:
    notes: DefaultDict[float, List[Note]] = defaultdict(list)
    current_beat = 0.0
    for measure in piece.measures:
        for part in measure.notes:
            for position, n in part.items():
                notes[current_beat + position] += n
        current_beat += measure.beats()
    
    notes_seq: List[Tuple[float, List[Note]]] = list(notes.items())
    notes_seq.sort(key=lambda n: n[0])

    max_voices = 8
    voices: List[List[Tuple[float, Note]]] = []
    for position, notelist in notes_seq:
        unused_notes = set(notelist)
        for voice in voices:
            # find closest note
            closest_note = None
            min_dist = 1e100
            for note in unused_notes:
                voice_last = voice[len(voice)-1]
                if voice_last[0] + voice_last[1].duration > position or \
                    abs(voice[0][1].pitch - note.pitch) > 12:
                    continue
                dist = (note.pitch - voice_last[1].pitch)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_note = note
                
            
            if closest_note is not None:
                voice.append((position, closest_note))
                unused_notes.remove(closest_note)

        # create new voice for each unused note
        for note in unused_notes:
            voices.append([(position, note)])

    used_voices: List[List[Tuple[float, Note]]] = []
    for voice in voices:
        if len(voice) >= 4:
            used_voices.append(voice)
    
    for voice in used_voices:
        for i, (position, note) in enumerate(voice):
            if i < len(voice)-1:
                duration = min(voice[i+1][0] - voice[i][0], voice[i][1].duration * 4)
            else:
                duration = note.duration*2
            voice[i] = (voice[i][0], Note(pitch=voice[i][1].pitch, duration=duration))

    used_voices.sort(key=lambda voice: sum(t[1].pitch for t in voice) / len(voice), reverse=True)

    used_voices_2: List[List[Tuple[float, Note]]] = []
    parts: List[str] = ["Piano"]
    for i, voice in enumerate(used_voices):
        min_pitch = min(t[1].pitch for t in voice)
        if min_pitch >= 55-21:
            parts.append("Violin")
            used_voices_2.append(voice)
        elif min_pitch >= 48-21:
            parts.append("Viola")
            used_voices_2.append(voice)
        else:
            parts.append("Cello")
            used_voices_2.append(voice)
                
        if min_pitch > 79-21:
            parts.append("Flute")
            used_voices_2.append(voice)
        elif i % 5 == 0:
            if min_pitch >= 68-21:
                parts.append("Oboe")
                used_voices_2.append(voice)
            elif min_pitch >= 55-21:
                parts.append("Clarinet")
                used_voices_2.append(voice)
            else:
                parts.append("Bassoon")
                used_voices_2.append(voice)

    orchestrated_measures: List[Measure] = []
    current_position = 0.0
    for measure in piece.measures:
        orchestrated_notes: List[NoteList] = [measure.notes[0]]
        for voice in used_voices_2:
            note_dict: DefaultDict[float, List[Note]] = defaultdict(list)
            for position, note in voice:
                if current_position <= position and position < current_position + measure.beats():
                    note_dict[position - current_position].append(note)
            orchestrated_notes.append(note_dict)
    
        orchestrated_measure = Measure(notes=orchestrated_notes, time_sig=measure.time_sig, tempo=measure.tempo)
        orchestrated_measures.append(orchestrated_measure)
        current_position += measure.beats()

    print(len(used_voices_2))
    print(len(parts))
    return Piece(measures=orchestrated_measures, parts=parts)