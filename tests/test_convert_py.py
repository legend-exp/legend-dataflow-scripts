from __future__ import annotations

import numpy as np
import pytest

from legenddataflowscripts.utils import convert_dict_np_to_float


@pytest.mark.parametrize(
    ("value", "expected_type", "expected_value"),
    [
        (np.float32(1.5), float, 1.5),
        (np.float64(2.5), float, 2.5),
        (np.int32(3), int, 3),
        (np.int64(4), int, 4),
        (np.bool_(True), bool, True),
        (np.bool_(False), bool, False),
    ],
)
def test_scalar_types(value, expected_type, expected_value):
    result = convert_dict_np_to_float({"k": value})
    assert type(result["k"]) is expected_type
    assert result["k"] == expected_value


def test_native_values_unchanged():
    d = {"a": 1, "b": 2.0, "c": "str", "d": None, "e": True}
    result = convert_dict_np_to_float(d)
    assert result == d
    for k, v in d.items():
        assert type(result[k]) is type(v)


def test_nested_dict():
    d = {"outer": {"inner": np.float64(1.0), "also": np.int32(2)}}
    result = convert_dict_np_to_float(d)
    assert type(result["outer"]["inner"]) is float
    assert type(result["outer"]["also"]) is int


def test_list_with_numpy_scalars():
    d = {"lst": [np.float64(1.0), np.int32(2), "keep", 3]}
    result = convert_dict_np_to_float(d)
    assert result["lst"] == [1.0, 2, "keep", 3]
    assert type(result["lst"][0]) is float
    assert type(result["lst"][1]) is int
    assert type(result["lst"][2]) is str
    assert type(result["lst"][3]) is int


def test_tuple_converted_to_list():
    d = {"t": (np.float64(1.0), 2)}
    result = convert_dict_np_to_float(d)
    assert result["t"] == [1.0, 2]


def test_returns_same_object():
    d = {"a": np.float64(1.0)}
    result = convert_dict_np_to_float(d)
    assert result is d


def test_deeply_nested():
    d = {"a": {"b": {"c": np.int64(42)}}}
    result = convert_dict_np_to_float(d)
    assert result["a"]["b"]["c"] == 42
    assert type(result["a"]["b"]["c"]) is int
