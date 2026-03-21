"""
Unit tests for Clinical Data Collection Protocol utilities.
"""

import pytest
from data.collection.protocol import (
    validate_demonstration_coverage,
    PHARMACEUTICAL_OBJECT_CATEGORIES
)


def test_validation_passes_with_complete_dataset():
    manifest = {cat: 15 for cat in PHARMACEUTICAL_OBJECT_CATEGORIES}
    manifest.update({'no_human': 10, 'human_stationary': 8,
                     'human_reaching_into_workspace': 10, 'human_moving_through_aisle': 6})
    result = validate_demonstration_coverage(manifest)
    assert result['valid'] is True
    assert len(result['coverage_gaps']) == 0


def test_validation_detects_category_gaps():
    manifest = {
        'small_rigid_bottle': 5,
        'blister_pack': 15,
        'capped_syringe': 3,
        'iv_bag': 12,
        'vial': 10,
        'medication_box': 11
    }
    result = validate_demonstration_coverage(manifest)
    assert result['valid'] is False
    gap_categories = [g['category'] for g in result['coverage_gaps']]
    assert 'small_rigid_bottle' in gap_categories
    assert 'capped_syringe' in gap_categories


def test_validation_passes_at_minimum():
    manifest = {cat: 10 for cat in PHARMACEUTICAL_OBJECT_CATEGORIES}
    result = validate_demonstration_coverage(manifest)
    assert result['valid'] is True


def test_total_demonstration_count():
    manifest = {cat: 10 for cat in PHARMACEUTICAL_OBJECT_CATEGORIES}
    result = validate_demonstration_coverage(manifest)
    assert result['total_demonstrations'] == 10 * len(PHARMACEUTICAL_OBJECT_CATEGORIES)
