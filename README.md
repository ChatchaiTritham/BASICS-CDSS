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
- Manuscript figure script:
  - `scripts/generate_manuscript_figures.py`: curated manuscript figure set
- Notebooks:
  - `notebooks/00_quickstart.ipynb`
  - `notebooks/01_basics_scenario_instantiation.ipynb`
  - `notebooks/02_basics_beyond_accuracy_metrics.ipynb`
  - `notebooks/03_coverage_risk_selective_prediction.ipynb`
  - `notebooks/04_harm_aware_evaluation.ipynb`
  - `notebooks/05_end_to_end_pipeline.ipynb`
  - `notebooks/06_digital_twin_simulation.ipynb`

## Curated Manuscript Figures

The curated manuscript figure set is maintained for manuscripts that are still
in preparation. This status does not imply publication, acceptance, or final
journal readiness for every raw baseline, demo, output, or legacy image in the
repository.

Regenerate the curated figure set:

```bash
python scripts/generate_manuscript_figures.py
```

Outputs:

- `figures/manuscript/`: selected PDF and PNG manuscript/supplementary figures
- `FIGURE_MANIFEST.csv`: figure role, source script, source artifact, caption,
  and intended article section

The broader `examples/figures/`, `examples/output/`, and `figures/legacy/`
trees remain reproducibility archives unless a figure is promoted into the
manifest.

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

## Manuscript Alignment

The manuscript frames BASICS-CDSS as a beyond-accuracy evaluation methodology
for clinical decision support. This repository supplies the corresponding code
and figure artifacts for:

- calibration and reliability analysis
- coverage-risk and selective prediction behavior
- abstention-aware evaluation
- harm-by-tier summaries
- decision-curve and net-benefit interpretation
- digital-twin style temporal evaluation

The manuscript currently uses local figure names, while this repository now
exports a curated manuscript/supplementary figure set under
`figures/manuscript/`. Treat `FIGURE_MANIFEST.csv` as the repository-side
source of truth for promoted figure artifacts.

## Contact

### Contact Author

**Chatchai Tritham** (Author)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- ORCID: [0000-0001-7899-228X](https://orcid.org/0000-0001-7899-228X)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- E-mail: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- ORCID: [0000-0003-4660-4590](https://orcid.org/0000-0003-4660-4590)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand
