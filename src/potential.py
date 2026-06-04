"""Well potential (entitlement) model — the deterministic backbone of deferment.

Deferment only means something against a *potential*: the rate a well WOULD make if
fully up and unconstrained. Estimating it without circular reasoning:

  1. Use only **full-uptime days** (runtime ≥ 90%) — days the well was essentially up,
     so their rate reflects capability, not downtime. (Down/partial days are excluded
     so they can't drag the estimate down.)
  2. Take a trailing **upper-ish quantile (P75)** of those up-day rates over a rolling
     window. P75 (not the median) biases toward the well's better days so a stretch of
     *curtailed-but-up* days can't quietly redefine capability; the rolling window lets
     capability decline naturally over time.

Pair this with the deadband in ``deferment.py`` (losses under ~8% of potential are
treated as measurement noise, not deferment) so a healthy well reads ~0 deferred.
Transparent and decline-aware — the kind of model an ops/reserves engineer will accept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 28      # trailing days for the capability estimate
DEFAULT_Q = 0.75         # quantile of up-day rates representing capability
UP_RUNTIME = 90.0        # a day at/above this runtime % counts toward capability


def well_potential(prod: pd.DataFrame, window: int = DEFAULT_WINDOW,
                   q: float = DEFAULT_Q) -> pd.Series:
    """Per-day potential (entitlement) oil rate, BOPD. Index aligns with ``prod``."""
    r = (prod["runtime_pct"].clip(lower=0, upper=100) / 100.0)
    up = prod["runtime_pct"] >= UP_RUNTIME
    # Full-time-equivalent rate on up days only (NaN elsewhere so they're ignored).
    rate_up = (prod["bopd"] / r.clip(lower=0.9, upper=1.0)).where(up)
    cap = rate_up.rolling(window=window, min_periods=4).quantile(q)
    cap = cap.ffill().bfill().fillna(prod["bopd"])
    # Potential can never be below what the well already produced that day.
    return pd.Series(np.maximum(cap.to_numpy(), prod["bopd"].to_numpy()), index=prod.index)
