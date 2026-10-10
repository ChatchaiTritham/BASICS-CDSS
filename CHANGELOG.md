# Changelog

All notable changes to this repository are documented here.

## Unreleased

### Revision 2026-10-11 (manuscript revision analyses)

- `run_all.py`: emits per-twin antibiotic-delay sweep (`counterfactual_delay_per_twin.csv`);
  BEWS computation factored into `_bews_from_groups`; `_initial_state` accepts
  `force_worsening` (default unchanged). Existing outputs are byte-identical.
- New: input-redundancy control (`redundancy_control.py`), mid-horizon-onset
  sensitivity (`onset_sensitivity.py`), counterfactual segment slopes
  (`counterfactual_segments.py`), noise-realisation averaging (`noise_realisations.py`),
  across-seed rank tests (`stat_tests.py`), label diagnostics (`label_diagnostics.py`),
  shared helper `revision_common.py`.
- Figures: counterfactual figure shows segment fits; robustness figure averages 30
  noise draws; new `fig_input_control`.

- Standardized root metadata, authorship, reference, testing, and container files.
- Verified local tests and cleaned generated cache or coverage artifacts before publishing.
