"""Per-disease distribution of the simulated mortality probability (results/cohort.csv).

Characterises the label generator: per disease, how many
twins the generator assigns a near-zero death probability (< 0.05) and how many of them
died. A predicted-low-risk group drawn from these twins has no events by construction.

    python scripts/label_diagnostics.py -> results/label_diagnostics.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402


def main():
    c = pd.read_csv(run_all.RESULTS_DIR / "cohort.csv")
    rows = []
    for dz, g in c.groupby("disease"):
        low = g[g["mortality_prob"] < 0.05]
        rows.append({"disease": dz, "n": len(g), "deaths": int(g["outcome"].sum()),
                     "n_p_death_lt_0.05": len(low), "deaths_p_death_lt_0.05": int(low["outcome"].sum()),
                     "median_p_death": g["mortality_prob"].median(),
                     "max_p_death": g["mortality_prob"].max()})
    out = pd.DataFrame(rows)
    out.to_csv(run_all.RESULTS_DIR / "label_diagnostics.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
