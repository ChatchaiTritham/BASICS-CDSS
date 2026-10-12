"""Sensitivity analysis: deterioration that begins mid-horizon instead of at presentation.

In the canonical cohort the outcome is driven by the worsening-vs-stable presentation
sub-population, so failing and surviving twins are separable at t=0. Here every twin
presents from the STABLE sub-population; a seeded 40% then receive a deterioration
insult at an onset hour drawn uniformly from {2, ..., 16}: at that step the observable
state is replaced by a draw from the disease's worsening sub-population (for the
cardiac model, whose latent ischaemia is carried rather than re-read from the vitals,
the latent ischaemia is re-seeded from the new ST-elevation / troponin with the same
formula the model uses at presentation). Everything downstream -- disease dynamics,
damage -> mortality link, Bernoulli label, 12-h observation window, degradation,
split, model families -- is the canonical pipeline.

The insult is an abrupt jump, not a physiological deterioration model; this is a
sensitivity setting for the evaluation diagnostics, not a new disease model.
Onsets after hour 12 fall outside the observation window by design.

Outputs
  results/onset_sensitivity_metrics.csv   seed x model x input x regime (AUROC / Brier / ECE)
  results/onset_sensitivity_summary.csv   per (model, input): static AUROC and AUROC drop
  results/onset_sensitivity_cohort.csv    per seed: prevalence overall / by onset group
  results/onset_sensitivity_bews.csv      BEWS (canonical definition) at the first seed

Usage:  python scripts/onset_sensitivity.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import revision_common as rc
from revision_common import run_all
from basics_cdss.temporal.disease_models import DiseaseModel

P_DETERIORATE = 0.4
ONSET_HOURS = np.arange(2, 17)  # inclusive 2..16


class _OnsetInsult(DiseaseModel):
    """Wrap a disease model; overwrite the observable state once at ``onset_step``."""

    def __init__(self, base, onset_step: int | None, insult: dict, disease: str):
        self.base, self.onset_step, self.insult, self.disease = base, onset_step, insult, disease
        self.k = 0

    def evolve(self, current_state, dt, interventions=None, rng=None):
        state = current_state
        if self.onset_step is not None and self.k == self.onset_step:
            state = dict(current_state)
            state.update(self.insult)
            if self.disease == "cardiac":
                state["_ischemia_severity"] = float(np.clip(
                    self.insult["st_elevation"] / 3.6 + self.insult["troponin"] / 40.0, 0, 1))
        self.k += 1
        return self.base.evolve(state, dt, interventions=interventions, rng=rng)


def build_onset_cohort(seed: int):
    """Return (X, X_seq, y, meta_df, per_disease_full_trajectories, n_full).

    ``X`` and ``X_seq`` keep NaN for disease-disjoint features; ``evaluate_inputs``
    imputes them with training-split medians."""
    rng = np.random.RandomState(seed)
    onset_rng = np.random.RandomState(seed + 31337)
    label_rng = np.random.RandomState(seed + 7919)
    models = {"sepsis": run_all.SepsisModel(), "ards": run_all.RespiratoryDistressModel(),
              "cardiac": run_all.CardiacEventModel()}
    cols = run_all._FEATURE_COLUMNS
    n_win = int(run_all.OBSERVATION_WINDOW_HOURS / run_all.DT) + 1
    n_full = int(run_all.HORIZON_HOURS / run_all.DT) + 1

    rows, seqs, per_disease = [], [], {dz: [] for dz in run_all.COHORT}
    for disease, count in run_all.COHORT.items():
        offset = sum(ord(c) for c in disease) * 1000
        for i in range(count):
            init = run_all._initial_state(disease, rng, force_worsening=False)
            deteriorates = onset_rng.random() < P_DETERIORATE
            onset = int(onset_rng.choice(ONSET_HOURS))
            insult = run_all._initial_state(disease, onset_rng, force_worsening=True)
            model = _OnsetInsult(models[disease], onset if deteriorates else None, insult, disease)
            twin = run_all.PatientDigitalTwin(archetype_id=f"{disease}_{i:04d}", initial_state=init,
                                              disease_model=model, seed=seed + offset + i)
            traj = twin.simulate(horizon_hours=run_all.HORIZON_HOURS, dt=run_all.DT, stochastic=True)
            dmg = float(traj[-1].features.get(run_all._DAMAGE_KEY[disease], 0.0))
            p_death = float(run_all._MORTALITY_MAP[disease](dmg))
            outcome = int(label_rng.random() < p_death)
            rows.append({**{c: init.get(c, np.nan) for c in cols}, "disease": disease,
                         "deteriorates": deteriorates, "onset_hour": onset if deteriorates else np.nan,
                         "mortality_prob": p_death, "outcome": outcome})
            seq = np.vstack([run_all._twin_timestep_features(s.features) for s in traj])
            seqs.append(seq[:n_win])
            per_disease[disease].append((outcome, {
                b: np.array([float(s.features.get(b, np.nan)) for s in traj])[:n_full]
                for b in run_all._DISEASE_BIOMARKERS[disease]}))

    df = pd.DataFrame(rows)
    X_seq = np.stack(seqs)  # NaNs kept; imputed with training medians downstream
    return (df[cols].to_numpy(dtype=float), X_seq, df["outcome"].to_numpy(dtype=int),
            df, per_disease, n_full)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary CSV from the existing metrics CSV")
    args = ap.parse_args()
    if args.summarise_only:
        rc.summarise_file(run_all.RESULTS_DIR / "onset_sensitivity_metrics.csv",
                          run_all.RESULTS_DIR / "onset_sensitivity_summary.csv")
        return

    rows, cohort_rows = [], []
    for k in range(args.seeds):
        seed = args.start + k
        print(f"[seed {seed}] ({k + 1}/{args.seeds})", flush=True)
        np.random.seed(seed)
        X, X_seq, y, meta, per_disease, n_full = build_onset_cohort(seed)
        det = meta["deteriorates"].to_numpy()
        early = det & (meta["onset_hour"].to_numpy() <= 12)
        late = det & (meta["onset_hour"].to_numpy() > 12)
        cohort_rows.append({
            "seed": seed, "n": len(y), "prevalence": float(y.mean()),
            "frac_deteriorating": float(det.mean()),
            "prevalence_stable": float(y[~det].mean()),
            "prevalence_onset_le12h": float(y[early].mean()),
            "prevalence_onset_gt12h": float(y[late].mean()),
            **{f"prevalence_{dz}": float(y[meta["disease"].to_numpy() == dz].mean())
               for dz in run_all.COHORT},
        })
        if k == 0:
            bews = run_all._bews_from_groups(per_disease, n_full)
            bews.insert(0, "seed", seed)
            bews.to_csv(run_all.RESULTS_DIR / "onset_sensitivity_bews.csv", index=False)
        rows += rc.evaluate_inputs(X, X_seq, y, seed, groups=meta["disease"].to_numpy())

    res = run_all.RESULTS_DIR
    pd.DataFrame(rows).to_csv(res / "onset_sensitivity_metrics.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(res / "onset_sensitivity_cohort.csv", index=False)
    _, summary = rc.summarise_drops(rows)
    summary.to_csv(res / "onset_sensitivity_summary.csv", index=False)
    print(pd.DataFrame(cohort_rows).mean(numeric_only=True).round(3).to_string())
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
