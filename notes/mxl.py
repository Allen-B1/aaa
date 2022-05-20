from collections import defaultdict
import xml.etree.ElementTree as ET
from typing import DefaultDict, List, Optional, Tuple, Sequence, Dict
from .note import Note, Measure, NoteList, Piece
import pretty_midi  # type: ignore

def parse_note(elem: ET.Element, divisions: float) -> Note:
    pitch_elem = elem.find("pitch")
    step = pitch_elem.find("step").text  # type: ignore
    octave = pitch_elem.find("octave").text  # type: ignore
    alter = None if pitch_elem.find("alter") is None else pitch_elem.find("alter").text  # type: ignore
    pitch_name = step + ('#' if alter == "1" else 'b' if alter == "-1" else "") + octave  # type: ignore
    pitch = pretty_midi.note_name_to_number(pitch_name)

    duration_int = int(elem.find("duration").text)  # type: ignore
    duration = duration_int / divisions
    return Note(pitch=pitch-21, duration=duration)


def parse_measure(elem: ET.Element, time_sig: Tuple[int, int], divisions: int) -> Tuple[NoteList, Tuple[int, int], Optional[float], Optional[float], int]:
    divisions = int(divisions if elem.find("attributes/divisions") is None
                    else elem.find("attributes/divisions").text) # type: ignore

    time_sig = (time_sig if elem.find("attributes/time") is None 
                    else (int(elem.find("attributes/time/beats").text), int(elem.find("attributes/time/beat-type").text))) # type: ignore

    notes: Dict[float, List[Note]] = dict()
    position = 0.0
    prev_duration = 0.0
    for child_elem in elem:
        # create note object
        # TODO: ties
        if child_elem.tag == "note":
            if child_elem.find("pitch") is not None:
                note = parse_note(child_elem, divisions)

                note_position = position
                if child_elem.find("chord") is not None:
                    note_position = position - prev_duration

                notes[note_position] = notes.get(note_position, [])
                notes[note_position].append(note)
            duration_int = int(child_elem.find("duration").text)  # type: ignore
            prev_duration = duration_int / divisions  # type: ignore

        # increase position
        if child_elem.tag in ("note", "forward", "backup") and child_elem.find("chord") is None:
            duration_int = int(child_elem.find("duration").text)  # type: ignore
            duration = duration_int / divisions
            if child_elem.tag == "backup":
                position -= duration
            else:
                position += duration

    tempos = elem.findall("direction/sound[@tempo]")
    first_tempo: Optional[float] = None
    final_tempo: Optional[float] = None
    if len(tempos) != 0:
        first_tempo = float(tempos[0].attrib['tempo']) # type: ignore
        final_tempo = float(tempos[len(tempos) - 1].attrib['tempo']) # type: ignore
    
    return notes, time_sig, first_tempo, final_tempo, divisions 

def parse_piece(root: ET.Element) -> Piece:
    part_elems = root.findall("part")

    # measure num => (part id => notes)
    note_lists: DefaultDict[int, Dict[int, NoteList]] = defaultdict(lambda: dict())
    time_sigs: Dict[int, Tuple[int, int]] = dict()
    first_tempos: Dict[int, float] = dict()
    final_tempos: Dict[int, float] = dict()

    for part_id, part_elem in enumerate(part_elems):
        measure_elems = part_elem.findall("measure")
        time_sig = (0, 0)
        divisions = 0
        for measure_elem in measure_elems:
            measure_num = int(measure_elem.attrib['number']) - 1 # starts at 1
            note_list, time_sig, first_tempo, final_tempo, divisions = parse_measure(measure_elem, time_sig, divisions)
            note_lists[measure_num][part_id] = note_list
            time_sigs[measure_num] = time_sig
            if first_tempo is not None:
                first_tempos[measure_num] = first_tempo
            if final_tempo is not None:
                final_tempos[measure_num] = final_tempo
    
    num_measures = max(note_lists.keys())
    tempo = 0.0
    measures: List[Measure] = []
    for measure_num in range(num_measures):
        if measure_num in first_tempos:
            tempo = first_tempos[measure_num]

        note_lists_seq: List[NoteList] = [note_lists[measure_num][part_id] for part_id in range(len(part_elems))]
        measure = Measure(note_lists_seq, time_sigs[measure_num], tempo)
        measures.append(measure)
    

    part_list: List[str] = [root.find("part-list/score-part[@id=\"" + part_elem.attrib['id'] + "\"]/score-instrument/instrument-name").text for part_elem in part_elems] # type: ignore

    return Piece(measures, parts=part_list)
    

def parse_file(filename: str) -> Piece:
    tree = ET.parse(filename)
    return parse_piece(tree.getroot())
