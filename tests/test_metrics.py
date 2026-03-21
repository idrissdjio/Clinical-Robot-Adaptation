"""
Unit tests for ClinBench-MedDel evaluation metrics.
"""

import pytest
from benchmark.clinbench_meddel.metrics import (
    task_success_rate, cross_environment_variance,
    adaptation_efficiency, human_interaction_safety
)


def test_task_success_rate_all_success():
    assert task_success_rate([{'success': True}] * 10) == 1.0


def test_task_success_rate_mixed():
    episodes = [{'success': True}] * 8 + [{'success': False}] * 2
    assert task_success_rate(episodes) == 0.8


def test_task_success_rate_empty():
    assert task_success_rate([]) == 0.0


def test_cross_environment_variance_identical():
    results = {'layout_a': 0.9, 'layout_b': 0.9, 'layout_c': 0.9}
    assert cross_environment_variance(results) == 0.0


def test_cross_environment_variance_varied():
    results = {'layout_a': 0.9, 'layout_b': 0.6}
    assert cross_environment_variance(results) > 0.0


def test_adaptation_efficiency_threshold_reached():
    result = adaptation_efficiency(150, 0.85, 0.8)
    assert result['threshold_reached'] is True


def test_adaptation_efficiency_threshold_not_reached():
    result = adaptation_efficiency(50, 0.6, 0.8)
    assert result['threshold_reached'] is False


def test_human_safety_no_collisions():
    episodes = [{'collision': False, 'near_miss': False}] * 20
    result = human_interaction_safety(episodes)
    assert result['collision_rate'] == 0.0
    assert result['safety_score'] == 1.0
