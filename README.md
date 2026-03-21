# ClinAdapt: Few-Shot Domain Adaptation for Autonomous Medication Delivery Robots

**Enabling national-scale deployment of robotic medication systems across diverse U.S. hospital environments through sample-efficient foundation model adaptation.**

---

## Overview

Robotic medication delivery systems have demonstrated clear clinical value — reducing nursing workload, decreasing medication handling errors, and improving hospital operational efficiency. Yet despite nearly a decade of commercial deployment, these systems operate in fewer than 0.5% of U.S. hospitals.

The barrier is not hardware. It is not demand. It is a specific, well-documented technical failure: **existing systems cannot generalize across hospital environments without expensive, time-consuming per-site retraining.** Each new hospital deployment requires thousands of demonstrations and weeks of engineering effort — making adoption economically inaccessible to the majority of U.S. hospitals, particularly rural, safety-net, and underserved facilities.

This project addresses that barrier directly.

**ClinAdapt** is a research initiative developing and validating a few-shot domain adaptation methodology for foundation model-based robotic manipulation, specifically applied to autonomous medication retrieval and delivery in hospital settings. The goal is to reduce the data and cost required to deploy robotic medication systems in a new hospital environment — from thousands of demonstrations and tens of thousands of dollars, to fewer than 200 demonstrations at a fraction of the cost.

---

## The Problem

### Why Robots Can't Scale Across Hospitals

Current robotic manipulation systems — including state-of-the-art foundation model-based approaches — rely on a **pretrain-then-finetune** paradigm. A system trained in Hospital A learns that environment's specific pharmacy layout, medication packaging configurations, lighting conditions, and staff interaction patterns. When moved to Hospital B, it fails or requires complete retraining.

As researchers from NYU, Meta, and Hello Robot confirmed at IEEE ICRA 2025:

> *"This need to finetune the foundation model for each and every new environment is limiting as it requires humans to collect data in the very environment where the robot is expected to perform. So while vision and language models have moved on to zero-shot deployments, such a capability eludes most robot manipulators."*
> — Etukuru et al., *Robot Utility Models*, IEEE ICRA 2025

This generalization failure has direct national consequences:
- **78,000** projected nursing deficit in the U.S. by 2025 (HRSA)
- **400,000+** preventable medication errors annually in U.S. hospitals (FDA)
- **<0.5%** of U.S. hospitals currently have access to robotic medication delivery systems
- **>67%** of AI healthcare deployments are misaligned with areas of greatest need (Hwang et al., Stanford, 2025)

### Why This Problem Is Hard

Hospital pharmacy environments present unique challenges that generic robotics research does not address:

1. **Pharmaceutical object diversity** — medications come in vials, blister packs, syringes, bottles, and pouches with varying fragility, weight, and surface properties requiring different grasp strategies
2. **Layout variability** — no two hospital pharmacies have the same shelf configurations, storage systems, or spatial constraints
3. **Dynamic human presence** — clinical staff move unpredictably through shared workspaces, requiring real-time trajectory adaptation rather than simple collision stopping
4. **Safety-critical constraints** — error rates must be near zero; a robot that drops or misidentifies a medication causes direct patient harm
5. **No standardized evaluation** — no benchmark currently exists defining what "clinically ready" generalization actually means in quantitative terms

---

## Proposed Contributions

This project produces three concrete, openly available deliverables:

### 1. Clinical Data Collection Protocol
A precise, reproducible specification of what demonstration data must be collected to adapt a robotic medication system to a new hospital environment. This protocol defines:
- Which pharmaceutical object categories must be represented
- Minimum demonstration counts per object class
- Required environmental variation coverage (lighting, shelf configurations, occlusion scenarios)
- Human interaction scenarios that must be captured
- Quality criteria for demonstration acceptance

*Informed by firsthand observation of robotic deployment failures in live clinical environments during the author's work at Diligent Robotics.*

### 2. Few-Shot Fine-Tuning Pipeline for Clinical Medication Environments
An adaptation pipeline built on the **Octo foundation model** — a transformer-based architecture for general-purpose robotic manipulation — validated on hospital pharmacy scenarios in simulation and physical hardware.

Target performance:
- Reliable task completion using **50-200 demonstrations** per new environment (vs. thousands in current approaches)
- Generalization across pharmaceutical object categories not seen during site-specific training
- Fluent human-aware trajectory adaptation when clinical staff enter the workspace

### 3. Clinical-Grade Evaluation Benchmark (ClinBench-MedDel)
The first standardized benchmark defining the quantitative performance thresholds a robotic medication system must meet for reliable deployment across diverse U.S. hospital environments.

ClinBench-MedDel will define:
- Task success rate thresholds across pharmaceutical object categories
- Cross-environment generalization metrics (performance variance across simulated hospital layouts)
- Human interaction safety metrics (collision frequency, trajectory smoothness near humans)
- Adaptation efficiency metrics (demonstrations required vs. performance achieved)

*This benchmark does not currently exist in the field. Without it, different teams optimize for different metrics that may not reflect clinical deployment requirements — contributing to the gap between laboratory results and real-world adoption.*

---

## Technical Approach

### Foundation Model: Octo

This project builds on **Octo** (Open X-Embodiment Team, 2024), an open-source generalist robot policy trained on the Open X-Embodiment dataset. Octo uses a transformer-based architecture that accepts multi-modal inputs (RGB images, language task descriptions, robot state) and predicts robot actions.

Key properties that make Octo suitable for clinical adaptation:
- Pre-trained on diverse manipulation data providing generalizable low-level skills
- Designed to be fine-tuned on domain-specific data with limited demonstrations
- Open-source with active research community support

### Adaptation Methodology

The core technical challenge is reducing the demonstration count required for reliable cross-environment generalization. This project investigates:

**Sample-efficient fine-tuning strategies:**
- Selective layer freezing to preserve generalizable skills while adapting environment-specific representations
- Data augmentation strategies for pharmaceutical object diversity
- Curriculum learning approaches that prioritize high-variance scenarios

**Human-aware control integration:**
- Real-time trajectory replanning when human presence is detected in the workspace
- Building on human-robot interaction research from the HIRO Laboratory at the University of Colorado Boulder
- Integration of force-torque sensing for safe manipulation near clinical staff

**Evaluation framework design:**
- Simulation environments in MuJoCo representing diverse hospital pharmacy layouts
- Pharmaceutical object mesh library covering common medication packaging types
- Standardized test protocols enabling reproducible comparison across adaptation approaches

---

## Repository Structure

```
clinadapt/
│
├── docs/
│   ├── clinical_data_collection_protocol_v0.1.md   # Draft protocol specification
│   ├── benchmark_design_notes.md                    # ClinBench-MedDel design rationale
│   └── literature_review.md                         # Relevant prior work
│
├── envs/
│   ├── pharmacy_sim/                                # MuJoCo pharmacy simulation environment
│   │   ├── assets/                                  # Shelf, medication object meshes
│   │   ├── configs/                                 # Hospital layout configurations
│   │   └── pharmacy_env.py                          # Gym-compatible environment wrapper
│   └── README.md
│
├── data/
│   ├── collection/
│   │   ├── protocol.py                              # Data collection utilities
│   │   └── validation.py                            # Demonstration quality checks
│   └── datasets/                                    # Collected demonstration data (gitignored)
│
├── models/
│   ├── octo_adapter/
│   │   ├── fine_tuning.py                           # Octo fine-tuning pipeline
│   │   ├── data_pipeline.py                         # Clinical dataset loading and preprocessing
│   │   └── evaluation.py                            # Performance evaluation utilities
│   └── baselines/                                   # Comparison baseline implementations
│
├── benchmark/
│   ├── clinbench_meddel/
│   │   ├── tasks/                                   # Benchmark task definitions
│   │   ├── metrics.py                               # Evaluation metric implementations
│   │   └── runner.py                                # Benchmark execution pipeline
│   └── results/                                     # Benchmark result logs
│
├── experiments/
│   ├── configs/                                     # Experiment configuration files
│   └── logs/                                        # Training and evaluation logs
│
├── scripts/
│   ├── setup_env.sh                                 # Environment setup
│   └── run_baseline_eval.sh                         # Baseline evaluation script
│
└── tests/
    └── test_env.py                                  # Environment and pipeline unit tests
```

---

## Current Status

This project is in active early development. Current progress:

- [x] Project scope and technical approach defined
- [x] Literature review completed (see `docs/literature_review.md`)
- [x] Draft clinical data collection protocol (see `docs/clinical_data_collection_protocol_v0.1.md`)
- [x] Repository structure and development environment established
- [ ] MuJoCo pharmacy simulation environment (in progress)
- [ ] Pharmaceutical object mesh library
- [ ] Baseline Octo fine-tuning pipeline
- [ ] Initial few-shot experiments on simulated environments
- [ ] ClinBench-MedDel benchmark v0.1
- [ ] Physical hardware validation (pending laboratory access)

---

## Background and Motivation

This project is led by **Idriss Djiofack Teledjieu**, a software engineer and robotics researcher based in the Boston area.

**Relevant experience:**
- Graduate Research Assistant, Human Interaction and Robotics (HIRO) Laboratory, University of Colorado Boulder (2023-2025) — hands-on research in foundation model adaptation (Octo) for robotic manipulation using the Franka Emika robotic arm
- Robotics Associate, Diligent Robotics (Summer 2024) — deployment and optimization of Moxi hospital service robots in clinical environments across the United States
- Software Engineer, MathWorks, Inc. (2025-present) — scalable software systems for simulation, control, and robotics applications

The core motivation for this project came directly from clinical observation: during hospital deployments at Diligent Robotics, the per-site variability of real clinical environments — different pharmacy layouts, different medication configurations, different staff workflows — was the primary operational challenge limiting adoption. The technical tools to address this challenge exist in the robotics research literature. The systematic, clinically-informed work of applying them rigorously to this context, and defining what success actually looks like in quantitative clinical terms, has not been done. That is what this project sets out to do.

---

## Clinical Impact Target

If successful, this work directly addresses:

| Problem | Scale | Source |
|---|---|---|
| U.S. nursing workforce deficit | 78,000 projected unfilled positions | HRSA, 2023 |
| Preventable hospital medication errors | 400,000+ annually | FDA |
| Hospital AI technology misalignment with need | >67% of deployments not reaching highest-need facilities | Hwang et al., Stanford, 2025 |
| Robotic medication system adoption | <0.5% of U.S. hospitals | Diligent Robotics / AHA data |

The evaluation benchmark produced by this project is designed to be openly available to any robotics team, hospital system, or manufacturer building clinical robotic systems — enabling the field to move from individual proprietary evaluations toward a shared standard for clinical deployment readiness.

---

## References

1. Etukuru, H., et al. (2025). *Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments.* IEEE ICRA 2025. NYU / Hello Robot / Meta.
2. Open X-Embodiment Collaboration. (2024). *Octo: An Open-Source Generalist Robot Policy.* arXiv:2405.12213.
3. Hwang, Y.M., et al. (2025). *AI Implementation in U.S. Hospitals: Regional Disparities and Health Equity Implications.* medRxiv. Stanford University.
4. Warmbein, A., et al. (2023). *Barriers and Facilitators in the Implementation of Mobilization Robots in Hospitals.* BMC Nursing, 22(45).
5. Goetz, L., et al. (2024). *Generalization — A Key Challenge for Responsible AI in Patient-Facing Clinical Applications.* npj Digital Medicine, 7(126). Cambridge / GSK.
6. Wessling, B. (October 28, 2025). *Moxi 2.0 Mobile Manipulator is Built for AI, Says Diligent Robotics.* The Robot Report.

---

## License

This project will be released under the MIT License. All benchmark definitions, evaluation protocols, and adaptation pipelines will be openly available for use by the research community, hospital systems, and robotics manufacturers.

---

## Contact

Idriss Djiofack Teledjieu
Framingham, MA
[GitHub] [LinkedIn] [Email]
