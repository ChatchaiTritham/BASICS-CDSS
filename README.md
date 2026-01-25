# BASICS-CDSS (Beyond Accuracy)

**BASICS-CDSS** (*Beyond Accuracy: Simulation-based Integrated Critical-Safety evaluation for Clinical Decision Support Systems*) is a **reproducible, simulation-based evaluation harness** for safety-critical clinical decision support.

This repository operationalizes the evaluation philosophy described in the manuscript:

> *Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems*.

It focuses on **pre-deployment behavioral safety under uncertainty** (e.g., escalation, abstention, calibration, harm-aware outcomes), and is intended as a **methodological and governance contribution**—not a claim of clinical effectiveness.

## What this repo provides
- **Archetype → Scenario instantiation** with controlled uncertainty (missingness/ambiguity/conflict)
- **Beyond-accuracy metrics**: calibration, coverage–risk, harm-aware scoring, explanation checks
- **Audit-friendly artifacts**: versioned configs, seeds, and exportable evidence tables/figures
- **Synthetic-only workflow** (no patient data)

## Related Projects

### SynDX (Synthetic Data Generation)

Scenarios are instantiated from **synthetic archetypes** provided by **SynDX**:

- Repository: [ChatchaiTritham/SynDX](https://github.com/ChatchaiTritham/SynDX)
- Purpose: Privacy-preserving synthetic medical data generation for vestibular disorders
- Features: 8,400 clinical archetypes, HL7 FHIR export, differential privacy

> **Note:** SynDX is used here as a *methodological test input* for stress-testing decision behavior. It is **not** presented as a clinically validated representation of patient populations.

### SAFE-Gate (Clinical Triage System)

BASICS-CDSS provides evaluation methodology that can be applied to systems like **SAFE-Gate**:

- Repository: [ChatchaiTritham/SAFE-Gate](https://github.com/ChatchaiTritham/SAFE-Gate)
- Purpose: Formally verified clinical decision support for emergency triage
- Features: 6-gate parallel architecture, provable safety guarantees, 95.3% sensitivity

## Quickstart
```bash
conda env create -f environment.yml
conda activate basics-cdss
jupyter lab
```

Open:
- `notebooks/00_quickstart.ipynb`
- `notebooks/01_basics_scenario_instantiation.ipynb`

## Notebooks
- **00** Quickstart end-to-end smoke run
- **01** Archetype → Scenario instantiation (Methods core)
- **02** Beyond-accuracy metrics (calibration, reliability)
- **03** Coverage–risk & abstention
- **04** Harm-aware evaluation
- **05** Explanation stability / consistency checks
- **06** Reporting pack export (tables + mapping)

## Package layout
- `basics_cdss.scenario` — archetype loader, instantiation, perturbations
- `basics_cdss.metrics` — calibration, coverage–risk, harm-aware metrics
- `basics_cdss.governance` — logging, reporting/export utilities

## How to Cite BASICS-CDSS
If you use this framework, codebase, or evaluation protocol, please cite:

Tritham C, Snae Namahoot C.  
**Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical Clinical Decision Support Systems.**  
*Healthcare Informatics Research.* (under review).

This repository provides a reproducible implementation of the **BASICS-CDSS** framework for simulation-based, pre-deployment evaluation of clinical decision support systems.

## Get in Touch

### Author

Chatchai Tritham

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

Chakkrit Snae Namahoot

- Email: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

## License

MIT (see `LICENSE`).
