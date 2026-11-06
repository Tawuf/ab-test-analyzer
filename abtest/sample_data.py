"""Generate the example experiment in data/checkout_experiment.csv.

Scenario: an online shop tests a simplified one-page checkout (variant B)
against the current multi-step checkout (A). The data is synthetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_checkout_experiment(n_users: int = 24_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variant = rng.choice(["A", "B"], size=n_users)
    base_rate = np.where(variant == "A", 0.048, 0.054)
    device = rng.choice(["mobile", "desktop"], size=n_users, p=[0.62, 0.38])
    rate = base_rate * np.where(device == "desktop", 1.25, 0.85)
    converted = (rng.random(n_users) < rate).astype(int)
    order_value = np.where(converted == 1, rng.lognormal(mean=6.4, sigma=0.55, size=n_users), 0.0)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "variant": variant,
            "device": device,
            "converted": converted,
            "revenue_sek": order_value.round(2),
        }
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "checkout_experiment.csv"
    make_checkout_experiment().to_csv(out, index=False)
    print(f"wrote {out}")
