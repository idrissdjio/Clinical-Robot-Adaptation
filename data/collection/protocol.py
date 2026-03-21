"""
ClinAdapt: Clinical Data Collection Protocol Utilities
Validates demonstration datasets against protocol requirements.
Status: Under development
"""

PHARMACEUTICAL_OBJECT_CATEGORIES = [
    'small_rigid_bottle',
    'blister_pack',
    'capped_syringe',
    'iv_bag',
    'vial',
    'medication_box'
]

MINIMUM_DEMONSTRATIONS_PER_CATEGORY = 10

HUMAN_PRESENCE_SCENARIOS = [
    'no_human',
    'human_stationary',
    'human_reaching_into_workspace',
    'human_moving_through_aisle'
]


def validate_demonstration_coverage(dataset_manifest: dict) -> dict:
    """
    Validate that a collected dataset meets minimum coverage requirements
    defined in the Clinical Data Collection Protocol v0.1.

    Args:
        dataset_manifest: Dict of object category -> demonstration count.

    Returns:
        dict: Validation results with coverage gaps identified.
    """
    gaps = []
    warnings = []

    for category in PHARMACEUTICAL_OBJECT_CATEGORIES:
        count = dataset_manifest.get(category, 0)
        if count < MINIMUM_DEMONSTRATIONS_PER_CATEGORY:
            gaps.append({
                'category': category,
                'collected': count,
                'required': MINIMUM_DEMONSTRATIONS_PER_CATEGORY,
                'deficit': MINIMUM_DEMONSTRATIONS_PER_CATEGORY - count
            })

    for scenario in HUMAN_PRESENCE_SCENARIOS:
        if scenario not in dataset_manifest:
            warnings.append(f'Missing human presence scenario: {scenario}')

    return {
        'valid': len(gaps) == 0,
        'coverage_gaps': gaps,
        'warnings': warnings,
        'total_demonstrations': sum(
            v for k, v in dataset_manifest.items()
            if k in PHARMACEUTICAL_OBJECT_CATEGORIES
        )
    }
