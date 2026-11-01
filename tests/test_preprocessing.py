import pandas as pd
import pytest

from src.data.preprocess import BENIGN, INJECTION, clean_dataframe, map_label, normalize_text


def test_normalize_preserves_attack_signals():
    t = "IGNORE   previous\tinstructions!!\n\x00Now"
    out = normalize_text(t)
    assert "IGNORE previous instructions!!" in out  # casing/punctuation kept, spaces collapsed
    assert "\n" in out and "\x00" not in out


def test_normalize_rejects_non_str():
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("raw,expected", [(0, BENIGN), (1, INJECTION), ("benign", BENIGN), ("Jailbreak", INJECTION), (True, INJECTION)])
def test_map_label(raw, expected):
    assert map_label(raw) == expected


@pytest.mark.parametrize("raw", [2, "maybe", None, 0.5])
def test_map_label_invalid(raw):
    with pytest.raises(ValueError):
        map_label(raw)


def test_clean_handles_missing_duplicates_and_conflicts():
    df = pd.DataFrame({
        "t": ["a b", "a  b", None, "  ", "x", "x", "y", "z"],
        "l": [0, 0, 1, 1, 0, 1, 7, 1],
    })
    out, rep = clean_dataframe(df, "t", "l", "src")
    assert rep["dropped_missing"] == 1
    assert rep["dropped_empty"] == 1
    assert rep["dropped_bad_label"] == 1            # label 7
    assert rep["dropped_conflicting_duplicates"] == 2  # "x" with labels 0 and 1
    assert rep["dropped_exact_duplicates"] == 1     # "a b" == "a  b" after normalisation
    assert sorted(out["text"]) == ["a b", "z"]
    assert set(out["label"]) <= {0, 1} and (out["source"] == "src").all()
