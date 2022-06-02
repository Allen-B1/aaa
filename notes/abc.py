from collections import defaultdict
import itertools
from typing import DefaultDict, Dict, List, Optional, Union
from .note import *
import pretty_midi
import fractions

def parse_pitch(abc: str, keysig: str) -> Tuple[Optional[int], int]:
    idx = 0
    # accidentals
    accidentals = None
    while abc[idx] in ["_", "=", "^"]:
        if accidentals is None: accidentals = 0
        accidentals += {
            "_": -1,
            "=": 0,
            "^": +1
        }[abc[idx]]
        idx += 1

    note_name = abc[idx]
    idx += 1
    assert note_name in "abcdefgzABCDEFG"
    if note_name == "z":
        return None, idx
    if note_name.islower():
        octave = 5
    else:
        octave = 4

    # octaves
    while len(abc) > idx and abc[idx] in ",'":
        octave += {",": -1, "'": +1}[abc[idx]]
        idx += 1

    if accidentals is None:
        sharp_keysigs = {
            "G": ["F"],
            "D": ["F", "C"],
            "A": ["F", "C", "G"],
            "E": ["F", "C", "G", "D"],
            "B": ["F", "C", "G", "D", "A"],
            "F#": ["F", "C", "G", "D", "A", "E"],
            "C#": ["F", "C", "G", "D", "A", "E", "B"]
        }

        flat_keysigs = {
            "F": ["B"],
            "Bb": ["B", "E"],
            "Eb": ["B", "E", "A"],
            "Ab": ["B", "E", "A", "D"],
            "Db": ["B", "E", "A", "D", "G"],
            "Gb": ["B", "E", "A", "D", "G", "C"],
            "Cb": ["B", "E", "A", "D", "G", "C", "F"]
        }

        mirrors = {
            "Abm": "Cb",
            "Am": "C",
            "Bbm": "Db",
            "Bm": "D",
            "Cm": "Eb",
            "C#m": "E",
            "Dm": "F",
            "Ebm": "Gb",
            "Em": "G",
            "Fm": "Ab",
            "F#m":"A",
            "Gm": "Bb",
            "G#m": "B",
        }
        if keysig in mirrors:
            keysig = mirrors[keysig]
        if keysig in sharp_keysigs:
            for raised_note in sharp_keysigs[keysig]:
                if note_name.lower() == raised_note.lower():
                    accidentals = 1
                    break
        elif keysig in flat_keysigs:
            for dep_note in flat_keysigs[keysig]:
                if note_name.lower() == dep_note.lower():
                    accidentals = -1
                    break
        
    if accidentals is None:
        accidentals = 0
    
    return pretty_midi.note_name_to_number(note_name.upper() + str(octave)) + accidentals - 21, idx


def parse_pitches(abc: str, keysig:str) -> Tuple[Optional[Sequence[int]], int]:
    if abc[0] == "[":
        idx = 1
        pitches: List[int] = []
        while True:
            if abc[idx] == "]":
                idx += 1
                break

            a, b = parse_pitch(abc[idx:], keysig)
            assert a is not None
            pitches.append(a)
            idx += b
        return pitches, idx
    else:
        a, b = parse_pitch(abc, keysig)
        if a is None:
            return a, b
        return [a], b

def parse_chord(abc: str, keysig: str, length: float) -> Tuple[Union[Sequence[Note], float], int]:
    idx = 0 
    # accents
    while abc[idx] in ["~", ".", "v", "u"]: idx += 1

    pitches, offset = parse_pitches(abc, keysig)
    idx += offset

    duration = length
    if len(abc) > idx and abc[idx] == "/":
        idx += 1
        duration *= 1 / int(abc[idx])
        idx += 1
    elif len(abc) > idx and abc[idx].isnumeric():
        duration *= int(abc[idx])
        idx += 1

    if pitches is None:
        return duration, idx
    else:
        return [Note(pitch=pitch, duration=duration) for pitch in pitches], idx

def parse_seq(abc: str, keysig: str, length: float) -> Tuple[NoteList, int]:
    idx = 0
    notelist: Dict[float, Sequence[Note]] = {}
    current_position = 0.0
    while idx < len(abc):
        if len(abc) <= idx:
            break

        # skip spaces
        while len(abc) > idx and abc[idx].isspace(): idx += 1

        if len(abc) <= idx:
            break

        notes, offset = parse_chord(abc[idx:], keysig, length)
        idx += offset

        if not isinstance(notes, float):
            notelist[current_position] = notes

            current_position += notes[0].duration
        else:
            current_position += notes

    return notelist, idx

def parse_piece(abc: str) -> Piece:
    """Parses a piece from a limited subset of ABC notation"""
    time_sig = (4, 4)
    key_sig = "C"
    beats_per_length = 1.0
    voice = 1
    measures: DefaultDict[int, List[NoteList]] = defaultdict(list)
    for line in abc.splitlines():
        if len(line.strip()) == 0: pass
        elif line.strip().startswith("M:"): # time signature
            t = line.split(":")[1].split("/")
            time_sig = (int(t[0].strip()), int(t[1].strip()))
        elif line.strip().startswith("L:"): # length
            beats_per_length = float(fractions.Fraction(line.split(":")[1].strip()) * 4)
        elif line.strip().startswith("K:"):
            key_sig = line.strip()[2:].strip()
        elif line.strip().startswith("V:"):
            voice = int(line.split(":")[1].strip().split(" ")[0].strip())
        elif line.strip()[1] == ":": pass
        else:
            seqs = list(filter(lambda x: x.strip() != "", line.split("|")))
            for seq in seqs:
                notes, _ = parse_seq(seq, key_sig, beats_per_length)
                measures[voice].append(notes)
        
    measures_list = list(measures.items())
    measures_list.sort(key=lambda t: t[0])
    real_measures: List[Measure] = []
    for i in itertools.count():
        has_measure = False
        notes_: List[NoteList] = []
        for voice, notelistlist in measures_list:
            if len(notelistlist) > i:
                has_measure = True
                notes_.append(notelistlist[i])
            else:
                notes_.append({})
        real_measures.append(Measure(notes=notes_, time_sig=time_sig, tempo=120))

        if not has_measure:
            break
    
    parts = ["Piano" for t in measures_list]
    return Piece(measures=real_measures, parts=parts)