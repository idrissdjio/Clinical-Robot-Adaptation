# Clinical Data Collection Protocol v0.1
## Adapting Robotic Medication Systems to New Hospital Environments

**Status:** Draft - Under Development  
**Version:** 0.1  
**Last Updated:** March 2026

---

## Purpose

This document defines the minimum viable demonstration dataset required to adapt a foundation model-based robotic manipulation system to a new hospital pharmacy environment. The goal is to specify what must be collected, how, and in what quantities to achieve reliable medication retrieval and delivery performance with minimum data collection burden.

This protocol is informed by firsthand clinical deployment experience observing where robotic medication systems fail when moved between hospital environments.

---

## Section 1: Pharmaceutical Object Categories

| Category | Examples | Key Challenge |
|---|---|---|
| Small rigid bottles | Oral medication bottles, eye drops | Precision grasp on smooth surfaces |
| Blister packs | Tablet strips, unit-dose packaging | Flat, flexible, varied sizes |
| Syringes (capped) | Pre-filled syringes | Small diameter, fragile |
| IV bags | Saline, medication infusion bags | Flexible, irregular shape |
| Vials | Injectable medication vials | Small, smooth, upright handling |
| Boxes | Packaged medications | Varied sizes, stable grasp |

Minimum coverage: At least 3 distinct objects per category.

---

## Section 2: Minimum Demonstration Counts (Draft)

| Scenario Type | Minimum Demonstrations |
|---|---|
| Standard retrieval per object category | 10-15 |
| Retrieval with shelf occlusion | 5-8 |
| Retrieval under variable lighting | 5 |
| Human presence - staff stationary | 5 |
| Human presence - staff reaching into workspace | 8-10 |
| Delivery handoff to station | 10 |

Estimated total: 50-80 demonstrations per new environment (to be validated experimentally).

---

## Section 3: Environmental Coverage Requirements

- [ ] At least 2 distinct shelf height configurations
- [ ] At least 2 distinct pharmacy layout orientations
- [ ] At least 2 lighting conditions
- [ ] At least 3 distinct staff presence scenarios
- [ ] Full pharmaceutical object category coverage

---

## Next Steps

- [ ] Validate minimum demonstration counts through simulation experiments
- [ ] Expand object category list based on clinical input
- [ ] Test protocol with synthetic data in MuJoCo simulation environment
