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
    
    # elongate notes
    for voice in used_voices:
        for i, (position, note) in enumerate(voice):
            if i < len(voice)-1:
                duration = min(voice[i+1][0] - voice[i][0], voice[i][1].duration * 4)
            else:
                duration = note.duration*2
            voice[i] = (voice[i][0], Note(pitch=voice[i][1].pitch, duration=duration))

    used_voices.sort(key=lambda voice: sum(t[1].pitch for t in voice) / len(voice), reverse=True)

    # string voices
    Voice = List[Tuple[float, Note]]
    voices_by_range: Tuple[List[Voice], List[Voice], List[Voice], List[Voice]] = ([], [], [], [])
    for i, voice in enumerate(used_voices):
        min_pitch = min(t[1].pitch for t in voice)
        if min_pitch >= 55-21:
            voices_by_range[0].append(voice)
        elif min_pitch >= 48-21:
            voices_by_range[1].append(voice)
        elif min_pitch >= 36-21:
            voices_by_range[2].append(voice)
        else:
            voices_by_range[3].append(voice)

    for voices in voices_by_range:
        voices.sort(key=lambda v: sum(1/t[1].duration for t in voice), reverse=True)

    # wind voices
    wind_voices: Dict[str, Voice] = {}
    for voice in voices_by_range[0]:
        min_pitch = min(t[1].pitch for t in voice)
        if min_pitch > 72-21 and not 'Flute' in wind_voices:
            wind_voices['Flute'] = voice
            continue
        if min_pitch >= 59-21 and not 'Oboe' in wind_voices:
            wind_voices['Oboe'] = voice
            continue
        
        if 'Oboe' in wind_voices and 'Flute' in wind_voices: break

    for voice in voices_by_range[1]:
        if min_pitch >= 55-21 and not 'Clarinet' in wind_voices:
            wind_voices['Clarinet'] = voice
            break
    
    for voice in voices_by_range[2]:
        if min_pitch >= 36-21 and not 'Bassoon' in wind_voices:
            wind_voices['Bassoon'] = voice
            break
    wind_voices_sorted = list(wind_voices.items())
    wind_voices_sorted.sort(key=lambda t: {
        "Flute": 0,
        "Oboe": 1,
        "Clarinet": 2,
        "Bassoon": 3}[t[0]])

    # aggregate together
    parts: List[str] = ["Piano"]
    for part_name, voice in wind_voices_sorted:
        parts.append(part_name)
    parts += ["Violin", "Violin", "Viola", "Cello", "Contrabass"]

    orchestrated_measures: List[Measure] = []
    current_position = 0.0
    for measure in piece.measures:
        orchestrated_notes: List[NoteList] = [measure.notes[0]] # piano first... of course...

        for instrument_name, voice in wind_voices_sorted: # winds
            note_dict: DefaultDict[float, List[Note]] = defaultdict(list)
            for position, note in voice:
                if current_position <= position and position < current_position + measure.beats():
                    note_dict[position - current_position].append(note)
            orchestrated_notes.append(note_dict)

        # violins (since violin 1 + 2)
        note_dict_1: DefaultDict[float, List[Note]] = defaultdict(list)
        note_dict_2: DefaultDict[float, List[Note]] = defaultdict(list)
        for i, voice in enumerate(voices_by_range[0]):
            for position, note in voice:
                if current_position <= position and position < current_position + measure.beats():
                    if i % 2 == 0:
                        note_dict_1[position - current_position].append(note)
                    else:
                        note_dict_2[position - current_position].append(note)
        orchestrated_notes += [note_dict_1, note_dict_2]

        for voices in voices_by_range[1:]: # viola, cello
            note_dict = defaultdict(list)
            for i, voice in enumerate(voices):
                for position, note in voice:
                    if current_position <= position and position < current_position + measure.beats():
                        note_dict[position - current_position].append(note)
            orchestrated_notes.append(note_dict)
    
        orchestrated_measure = Measure(notes=orchestrated_notes, time_sig=measure.time_sig, tempo=measure.tempo)
        orchestrated_measures.append(orchestrated_measure)
        current_position += measure.beats()

    print(len(parts))
    print(len(orchestrated_measures[0].notes))
    print(len(orchestrated_measures))
    return Piece(measures=orchestrated_measures, parts=parts)