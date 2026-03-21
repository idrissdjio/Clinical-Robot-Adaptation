# ClinBench-MedDel: Clinical Benchmark for Medication Delivery Robots
## Design Notes v0.1

**Status:** Early design phase  
**Last Updated:** March 2026

---

## Motivation

No standardized benchmark currently exists for evaluating whether a robotic medication delivery system is ready for deployment across diverse hospital environments. ClinBench-MedDel fills this gap.

---

## Proposed Metric Categories

### Task Success
- Medication retrieval success rate per object category
- Delivery completion rate
- Overall task completion rate across mixed scenarios

### Cross-Environment Generalization
- Performance variance across N simulated hospital layouts
- Performance drop from training to novel environment
- Minimum demonstrations to reach threshold performance

### Human Interaction Safety
- Collision frequency with clinical staff present
- Trajectory smoothness near human presence
- Task completion rate maintained under human presence

### Adaptation Efficiency
- Demonstrations required to reach 80% task success in new environment
- Demonstrations required to reach 90% task success
- Training time per new environment

---

## Proposed Threshold Values (Draft)

| Metric | Minimum | Target |
|---|---|---|
| Task success rate | 80% | 90% |
| Cross-environment variance | < 15% std dev | < 10% std dev |
| Collision rate with human present | < 1% | 0% |
| Demonstrations to 80% success | < 200 | < 100 |

All values preliminary, subject to experimental validation.

---

## Next Steps

- [ ] Implement MuJoCo simulation environment
- [ ] Define pharmaceutical object mesh library
- [ ] Implement metric calculation pipeline
- [ ] Run baseline Octo evaluation for reference numbers
