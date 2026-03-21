"""
ClinBench-MedDel: Evaluation Metrics
Clinical-grade evaluation metrics for robotic medication delivery systems.
Status: Under development
"""

import numpy as np
from typing import List, Dict


def task_success_rate(episodes: List[Dict]) -> float:
    """Calculate overall task success rate across evaluation episodes."""
    if not episodes:
        return 0.0
    return sum(ep['success'] for ep in episodes) / len(episodes)


def cross_environment_variance(layout_results: Dict[str, float]) -> float:
    """Performance variance across hospital layouts. Low variance = robust generalization."""
    rates = list(layout_results.values())
    if len(rates) < 2:
        return 0.0
    return float(np.std(rates))


def adaptation_efficiency(demonstrations_used: int, success_rate: float,
                          target_threshold: float = 0.8) -> Dict:
    """Evaluate demonstrations required to reach target performance threshold."""
    return {
        'demonstrations_used': demonstrations_used,
        'success_rate': success_rate,
        'target_threshold': target_threshold,
        'threshold_reached': success_rate >= target_threshold,
        'demonstrations_per_success_point': demonstrations_used / max(success_rate, 1e-6)
    }


def human_interaction_safety(episodes: List[Dict]) -> Dict:
    """Evaluate safety metrics during human-robot interaction scenarios."""
    if not episodes:
        return {'collision_rate': 0.0, 'near_miss_rate': 0.0, 'safety_score': 1.0}
    collision_rate = sum(ep.get('collision', False) for ep in episodes) / len(episodes)
    near_miss_rate = sum(ep.get('near_miss', False) for ep in episodes) / len(episodes)
    return {
        'collision_rate': collision_rate,
        'near_miss_rate': near_miss_rate,
        'safety_score': 1.0 - collision_rate
    }
