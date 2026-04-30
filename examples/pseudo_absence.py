"""Generate pseudo-absences for a small synthetic example."""

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from abil.pseudo_generation import generate_pseudo_absences

os.chdir(os.path.join(".", "examples"))

species = "synthetic_species"
env_vars = ["temperature", "silicate"]
rng = np.random.default_rng(42)

missing_rows = pd.DataFrame(
    {
        "temperature": rng.uniform(0, 25, 1200),
        "silicate": rng.uniform(0.1, 3, 1200),
    }
)

presence_pool = missing_rows[
    (missing_rows["temperature"] > 5) & (missing_rows["silicate"] < 1)
]
merged_df = presence_pool.sample(n=150, random_state=42).copy()
merged_df[species] = 1
missing_rows = missing_rows.drop(index=merged_df.index)

augmented = generate_pseudo_absences(
    merged_df,
    missing_rows,
    env_vars,
    [species],
    absence_ratio=1,
    min_presence=50,
)
pseudo_absences = augmented[augmented[species] == 0]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    missing_rows["temperature"],
    missing_rows["silicate"],
    alpha=0.15,
    s=12,
    label="Candidate rows",
)
ax.scatter(
    merged_df["temperature"],
    merged_df["silicate"],
    marker="x",
    s=28,
    label="Observed presences",
)
ax.scatter(
    pseudo_absences["temperature"],
    pseudo_absences["silicate"],
    alpha=0.85,
    s=45,
    label="Pseudo-absences",
)
ax.set_xlabel("temperature")
ax.set_ylabel("silicate")
ax.set_title("Pseudo-absences in environmental space")
ax.legend()
fig.tight_layout()
plt.savefig("pseudo_absence.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
