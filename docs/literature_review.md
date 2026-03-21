# Literature Review: Few-Shot Domain Adaptation for Clinical Robotic Manipulation

## Core Problem References

1. **Etukuru et al. (2025)** - Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments. IEEE ICRA 2025. NYU / Hello Robot / Meta.
   - Key finding: Per-environment fine-tuning is the primary barrier preventing robotic manipulation from achieving scalable deployment. 'While vision and language models have moved on to zero-shot deployments, such a capability eludes most robot manipulators.'

2. **Open X-Embodiment Team (2024)** - Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.
   - Key finding: Transformer-based foundation model for robotic manipulation, designed for few-shot domain-specific fine-tuning. Base model for this project.

3. **Warmbein et al. (2023)** - Barriers and Facilitators in the Implementation of Mobilization Robots in Hospitals. BMC Nursing, 22(45).
   - Key finding: Non-adaptability to different conditions of the patient population is a documented barrier to hospital robot adoption.

4. **Goetz et al. (2024)** - Generalization: A Key Challenge for Responsible AI in Patient-Facing Clinical Applications. npj Digital Medicine, 7(126). Cambridge / GSK.
   - Key finding: Generalization is a major challenge in clinical AI. Recommends foundation model fine-tuning on scarce data as the promising direction.

5. **Hwang et al. (2025)** - AI Implementation in U.S. Hospitals: Regional Disparities and Health Equity Implications. medRxiv. Stanford University.
   - Key finding: Over 67% of AI healthcare deployments misaligned with areas of greatest need.

6. **Wessling (2025)** - Moxi 2.0 Mobile Manipulator is Built for AI. The Robot Report, Oct 2025.
   - Key finding: Moxi operates in fewer than 30 of ~6,000 U.S. hospitals despite a decade of deployment. CEO confirms hospital environment complexity is the core challenge.

## Gaps Identified

- No standardized benchmark exists for clinical-grade generalization of robotic medication systems.
- Existing few-shot robot learning work does not address pharmaceutical object constraints.
- Per-environment data collection protocols for clinical robotics have not been systematically defined.
