"""Simulator verification checks: step-size convergence and untreated severity course.

1. Step-size convergence. The canonical cohort (seed 42) is re-simulated with the
   step size dt = 1, 0.5 and 0.25 h, with (a) the noise switched off, so only the
   drift is discretised, and (b) the noise as implemented. Note that the committed
   models scale the per-step noise SD by dt (not sqrt(dt)), so in (b) the noise
   variance per hour shrinks with dt; (a) is the clean convergence check. For each
   twin the terminal latent damage is mapped to p(death); the label uses the same
   seeded uniform draw per twin as ``build_cohort``. Reported per (dt, noise mode,
   disease): mean p(death), prevalence, and -- against dt = 0.25 h -- the mean and
   maximum absolute difference in p(death) and the label agreement.

2. Untreated severity course (sepsis). The fraction of sepsis twins whose latent
   infection severity S exceeds the damage threshold theta_D (S > theta_D means
   organ damage is still accruing) at 0, 3, 6, 9, 12, 18 and 24 h: (a) in the
   canonical, untreated cohort; (b) in the 120 twins of the antibiotic-delay sweep,
   in the arm treated at 18 h (untreated up to 18 h).

    python scripts/simulator_checks.py -> results/dt_convergence.csv,
                                          results/severity_course.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

DTS = (1.0, 0.5, 0.25)
CHECK_HOURS = (0, 3, 6, 9, 12, 18, 24)


class _ZeroNoise:
    """RNG stand-in whose normal draws are all zero (drift only)."""

    def normal(self, loc=0.0, scale=1.0, size=None):
        return 0.0 if size is None else np.zeros(size)


class _Drift(run_all.DiseaseModel if hasattr(run_all, "DiseaseModel") else object):
    def __init__(self, base):
        self.base = base

    def evolve(self, current_state, dt, interventions=None, rng=None):
        return self.base.evolve(current_state, dt, interventions=interventions, rng=_ZeroNoise())


def _cohort_p_death(dt: float, noise: bool) -> pd.DataFrame:
    rng = np.random.RandomState(run_all.SEED)
    label_rng = np.random.RandomState(run_all.SEED + 7919)
    models = run_all._disease_models()
    rows = []
    for disease, count in run_all.COHORT.items():
        offset = sum(ord(c) for c in disease) * 1000
        model = models[disease] if noise else _Drift(models[disease])
        for i in range(count):
            init = run_all._initial_state(disease, rng)
            twin = run_all.PatientDigitalTwin(archetype_id=f"{disease}_{i:04d}", initial_state=init,
                                              disease_model=model, seed=run_all.SEED + offset + i)
            traj = twin.simulate(horizon_hours=run_all.HORIZON_HOURS, dt=dt, stochastic=True)
            dmg = float(traj[-1].features.get(run_all._DAMAGE_KEY[disease], 0.0))
            p = float(run_all._MORTALITY_MAP[disease](dmg))
            u = label_rng.random()
            rows.append({"disease": disease, "twin": i, "p_death": p, "outcome": int(u < p)})
    return pd.DataFrame(rows)


def dt_convergence() -> pd.DataFrame:
    out = []
    for noise in (False, True):
        runs = {dt: _cohort_p_death(dt, noise) for dt in DTS}
        ref = runs[min(DTS)]
        for dt, r in runs.items():
            for dz in list(run_all.COHORT) + ["all"]:
                m = np.ones(len(r), bool) if dz == "all" else (r["disease"] == dz).to_numpy()
                diff = np.abs(r["p_death"].to_numpy()[m] - ref["p_death"].to_numpy()[m])
                out.append({
                    "noise": "as_implemented" if noise else "off", "dt_hours": dt,
                    "disease": dz, "n": int(m.sum()),
                    "mean_p_death": float(r["p_death"][m].mean()),
                    "prevalence": float(r["outcome"][m].mean()),
                    "mean_abs_diff_p_vs_dt0.25": float(diff.mean()),
                    "max_abs_diff_p_vs_dt0.25": float(diff.max()),
                    "label_agreement_vs_dt0.25": float(
                        (r["outcome"].to_numpy()[m] == ref["outcome"].to_numpy()[m]).mean()),
                })
    return pd.DataFrame(out)


def severity_course() -> pd.DataFrame:
    theta = run_all._disease_models()["sepsis"].theta_damage
    idx = [int(h / run_all.DT) for h in CHECK_HOURS]
    rows = []

    # (a) canonical untreated cohort
    cohort = run_all.build_cohort()
    sep = cohort[cohort.disease == "sepsis"].reset_index(drop=True)
    S = []
    for disease, i, init, traj in run_all._simulate_cohort_trajectories():
        if disease == "sepsis":
            S.append([float(traj[k].features.get("_infection_severity", np.nan)) if k > 0
                      else np.nan for k in idx])
    S = np.array(S)
    for j, h in enumerate(CHECK_HOURS):
        if h == 0:
            continue  # severity is first written at the end of the first step
        above = S[:, j] > theta
        rows.append({"population": "canonical_sepsis_untreated", "hour": h, "n": len(S),
                     "theta_damage": theta, "frac_S_above_theta": float(above.mean()),
                     "frac_above_among_deaths": float(above[sep["outcome"] == 1].mean()),
                     "frac_above_among_survivors": float(above[sep["outcome"] == 0].mean())})

    # (b) delay-sweep twins, arm treated at 18 h (untreated through 18 h)
    rng = np.random.RandomState(run_all.SEED)
    model = run_all._disease_models()["sepsis"]
    S2 = []
    for i in range(120):
        init = run_all._initial_state("sepsis", rng)
        init["temperature"] = max(init["temperature"], 38.3)
        init["lactate"] = max(init["lactate"], 2.2)
        twin = run_all.PatientDigitalTwin(archetype_id=f"sepsis_cf_{i:04d}", initial_state=init,
                                          disease_model=model, seed=run_all.SEED + i)
        traj = twin.simulate(horizon_hours=run_all.HORIZON_HOURS, dt=run_all.DT,
                             intervention_schedule={18.0: {"antibiotic": True}}, stochastic=True)
        S2.append([float(traj[k].features.get("_infection_severity", np.nan)) if k > 0
                   else np.nan for k in idx])
    S2 = np.array(S2)
    for j, h in enumerate(CHECK_HOURS):
        if h == 0:
            continue
        rows.append({"population": "delay_sweep_twins_treated_at_18h", "hour": h, "n": len(S2),
                     "theta_damage": theta, "frac_S_above_theta": float((S2[:, j] > theta).mean()),
                     "frac_above_among_deaths": np.nan, "frac_above_among_survivors": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(run_all.SEED)
    dtc = dt_convergence()
    dtc.to_csv(run_all.RESULTS_DIR / "dt_convergence.csv", index=False)
    sev = severity_course()
    sev.to_csv(run_all.RESULTS_DIR / "severity_course.csv", index=False)
    print(dtc.round(4).to_string())
    print(sev.round(3).to_string())


if __name__ == "__main__":
    main()
