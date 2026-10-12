#!/usr/bin/env bash
# Run order for every script behind the manuscript; prints wall time per step.
set -e
PY=${PY:-python}
cd "$(dirname "$0")/.."
for s in run_all multiseed redundancy_control onset_sensitivity bootstrap_ci noise_realisations \
         counterfactual_segments label_diagnostics simulator_checks stat_tests sensitivity \
         generate_results_figures make_extra_figs; do
  t0=$(date +%s); $PY scripts/$s.py > /dev/null
  echo "$s $(( $(date +%s) - t0 ))s"
done
