import numpy as np
import pandas as pd

def generate_synthetic(rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    txn_amount = rng.gamma(shape=2.0, scale=120.0, size=rows)
    txn_count_24h = rng.poisson(lam=4.0, size=rows)
    account_age_days = rng.integers(0, 2000, size=rows)
    is_international = rng.binomial(1, p=0.12, size=rows)
    device_change_7d = rng.binomial(1, p=0.10, size=rows)
    failed_logins_24h = rng.poisson(lam=0.8, size=rows)

    base = (
        0.0025 * txn_amount +
        0.12 * txn_count_24h +
        0.0009 * (2000 - account_age_days) +
        1.1 * is_international +
        0.9 * device_change_7d +
        0.35 * failed_logins_24h
    )
    noise = rng.normal(0, 1.2, size=rows)
    logits = base + noise - 3.0
    prob = 1 / (1 + np.exp(-logits))
    label = rng.binomial(1, p=np.clip(prob, 0, 1), size=rows)

    df = pd.DataFrame({
        "txn_amount": txn_amount.round(2),
        "txn_count_24h": txn_count_24h,
        "account_age_days": account_age_days,
        "is_international": is_international,
        "device_change_7d": device_change_7d,
        "failed_logins_24h": failed_logins_24h,
        "label": label
    })
    return df
