import xml.etree.ElementTree as ET
from typing import List, Tuple, Sequence, Dict
from .note import Note, Measure, Part
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


def parse_measure(elem: ET.Element, divisions: int) -> Tuple[Measure, int]:
    divisions = int(divisions if elem.find("attributes/divisions") is None
                    else elem.find("attributes/divisions").text)  # type: ignore
    notes: Dict[float, List[Note]] = dict()
    position = 0.0
    max_position = 0.0
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
        if position > max_position:
            max_position = position
    return Measure(notes, beats=max_position), divisions


def parse_part(elem: ET.Element) -> Part:
    measures_elem = elem.findall("measure")
    measures: List[Measure] = []
    divisions = 0
    for measure_elem in measures_elem:
        measure, divisions = parse_measure(measure_elem, divisions)
        measures.append(measure)
    return Part(measures=measures, instrument="piano")


def parse_tree(root: ET.Element) -> Sequence[Part]:
    parts: List[Part] = []

    parts_elem = root.findall("part")
    for part in parts_elem:
        parts.append(parse_part(part))
    return parts


def parse_file(filename: str) -> Sequence[Part]:
    tree = ET.parse(filename)
    return parse_tree(tree.getroot())
