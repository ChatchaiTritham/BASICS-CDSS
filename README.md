# BASICS-CDSS

## Overview

Simulation-based evaluation framework for safety-critical clinical decision support systems.

## Installation

```bash
pip install -e .
```

## Repository Structure

- `src/basics_cdss/`: source package
- `tests/`: automated tests
- `examples/`: example usage
- `notebooks/`: research notebooks

## Tutorials And Demos

- Example scripts:
  - `examples/visualization_demo.py`: visualization-oriented walkthrough
  - `examples/generate_performance_figures.py`: performance figure generation
  - `examples/generate_xai_figures.py`: explainability figure generation
  - `examples/generate_clinical_metrics_figures.py`: clinical metrics figure generation
  - `examples/publication_figures.py`: publication-style figure bundle
- Notebooks:
  - `notebooks/00_quickstart.ipynb`
  - `notebooks/01_basics_scenario_instantiation.ipynb`
  - `notebooks/02_basics_beyond_accuracy_metrics.ipynb`
  - `notebooks/03_coverage_risk_selective_prediction.ipynb`
  - `notebooks/04_harm_aware_evaluation.ipynb`
  - `notebooks/05_end_to_end_pipeline.ipynb`
  - `notebooks/06_digital_twin_simulation.ipynb`

## Cross-Repository Tutorial Charts

- `../tutorial_surface_comparison.png`: scripts vs examples vs notebooks across all repositories
- `../tutorial_asset_density.png`: interactive/tutorial asset density normalized by repository size
- `../tutorial_maturity_report.md`: combined maturity summary

## Source Layout

This repository uses the recommended `src/<package_name>` layout.
Importable code lives in `src/basics_cdss/`.

## Testing

```bash
pytest tests -v
```

## Contact

### Contact Author

**Chatchai Tritham** (PhD Candidate)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- Email: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- Department of Computer Science
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand
