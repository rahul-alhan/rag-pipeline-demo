"""Config invariants — guards against silent threshold drift."""
from __future__ import annotations

from src.config import CONFIG


def test_chunk_overlap_smaller_than_chunk_size():
    assert 0 <= CONFIG.chunk_overlap < CONFIG.chunk_size


def test_quality_gates_in_unit_interval():
    assert 0.0 < CONFIG.faithfulness_gate <= 1.0
    assert 0.0 < CONFIG.precision_gate <= 1.0


def test_mmr_lambda_in_range():
    assert 0.0 <= CONFIG.mmr_lambda <= 1.0


def test_top_k_is_positive():
    assert CONFIG.top_k >= 1


def test_collection_and_persist_dir_set():
    assert CONFIG.collection_name
    assert CONFIG.persist_dir
