"""Unit tests for the deterministic reasoning pieces."""
import pytest

from app.agents.ocr_agent import classify
from app.agents.voice_agent import parse_intent
from app.utils.floor import infer_floor_from_room, parse_floor_indicator, parse_room_range
from app.utils.geo import clock_from_bearing, clock_from_image_x, decode_polyline, distance_to_polyline_m
from app.utils.text import canonical_room, room_match_score


@pytest.mark.parametrize("room,floor", [("443B", 4), ("205A", 2), ("1205", 12), ("B105", 1), ("G12", 0),
                                        ("Room 312", 3), ("rm 4-21", 4)])
def test_floor_from_room(room, floor):
    assert infer_floor_from_room(room).floor == floor


def test_floor_explanation_mentions_reason():
    assert "floor 4" in infer_floor_from_room("443B").explanation


@pytest.mark.parametrize("text,floor", [("Level 3", 3), ("Floor 2", 2), ("4F", 4), ("L2", 2), ("G", 0),
                                        ("Ground", 0), ("B1", -1), ("3rd floor", 3), ("second floor", 2),
                                        ("Room 443B", None), ("Exit", None)])
def test_floor_indicator(text, floor):
    assert parse_floor_indicator(text) == floor


def test_room_canonicalisation_and_ocr_noise():
    assert canonical_room("Room 443-b") == "443B"
    assert canonical_room("4O3") == "403"           # O read for 0
    assert room_match_score("443B", "443B") == 1.0
    assert room_match_score("4438", "443B") >= 0.85  # B read as 8
    assert room_match_score("443A", "443B") < 0.85   # different room
    assert room_match_score("205A", "443B") < 0.5


@pytest.mark.parametrize("text,kind,value", [
    ("443B", "room", "443B"), ("Room 205A", "room", "205A"), ("Building C", "building", "C"),
    ("Gate 4", "gate", "4"), ("Level 4", "floor", "4"), ("Rooms 400-450 →", "direction", "to the right"),
    ("EXIT", "other", None),
])
def test_ocr_classification(text, kind, value):
    k, v, _ = classify(text)
    assert (k, v) == (kind, value)


def test_room_range():
    assert parse_room_range("Rooms 440 – 460") == (440, 460)


@pytest.mark.parametrize("utterance,intent", [
    ("Take me to Room 443B", "navigate"), ("Where am I?", "where_am_i"),
    ("What is in front of me?", "describe"), ("Guide me to reception.", "find"),
    ("Find elevator", "find"), ("find the nearest lift", "find"), ("Read the sign", "read_text"),
    ("stop", "stop"), ("Say that again", "repeat"), ("Is it busy here?", "chat"),
    ("Find room 205A", "navigate"),
])
def test_intents(utterance, intent):
    assert parse_intent(utterance).name == intent


def test_navigate_parses_room_and_building():
    d = parse_intent("Take me to Room 443B in Bush House").destination
    assert d.room == "443B" and d.place_query == "Bush House" and d.floor_hint == 4


def test_navigate_gate_and_building_letter():
    assert parse_intent("Take me to Gate 4").destination.sign_text == "Gate 4"
    d = parse_intent("Take me to Building C at Imperial College").destination
    assert d.sign_text == "Building C" and d.place_query == "Building C, Imperial College"


def test_find_target_synonyms():
    assert parse_intent("where is the front desk").target_kind == "reception"
    assert parse_intent("take me to the nearest toilet").target_kind == "restroom"


def test_clock_directions():
    assert clock_from_image_x(0.5) == 12
    assert clock_from_image_x(0.95) == 1
    assert clock_from_image_x(0.05) == 11
    assert clock_from_bearing(90, 0) == 3
    assert clock_from_bearing(0, 90) == 9
    assert clock_from_bearing(180, 0) == 6


def test_polyline():
    pts = decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    assert pts[0] == (38.5, -120.2) and len(pts) == 3
    assert distance_to_polyline_m(38.5, -120.2, pts) < 1


def test_precise_distance_and_object_phrases():
    from app.utils.geo import object_phrase, spoken_distance

    assert spoken_distance(1.24) == "1.2 meters"        # one decimal where a step matters
    assert spoken_distance(0.1) == "0.3 meters"         # floor: never "0.0"
    assert spoken_distance(2.96) == "3.0 meters"
    assert spoken_distance(4.4) == "4 meters"
    assert spoken_distance(37) == "35 meters"
    assert spoken_distance(487) == "490 meters"
    assert spoken_distance(1530) == "1.5 kilometers"
    assert spoken_distance(None) == ""
    assert object_phrase("chair", 1, "right", 2.4) == "Chair at 1 o'clock, 2.4 meters"
    assert object_phrase("door", None, "left", None) == "Door on your left"
    for m in (0.5, 1.4, 6, 60, 600):
        assert "about" not in spoken_distance(m) and "close" not in spoken_distance(m)
