"""Basin-level daily runoff metrics in physical units."""

import numpy as np
import pandas as pd


def _paired_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    observed = frame["y_obs"].to_numpy(dtype=float)
    simulated = frame["y_sim"].to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(simulated)
    return observed[valid], simulated[valid]


def _nse(observed, simulated) -> float:
    if observed.size < 2:
        return np.nan
    variance = np.sum((observed - observed.mean()) ** 2)
    if variance == 0:
        return np.nan
    return float(1.0 - np.sum((simulated - observed) ** 2) / variance)


def _kge(observed, simulated) -> float:
    if observed.size < 2:
        return np.nan
    mean_obs = observed.mean()
    std_obs, std_sim = observed.std(), simulated.std()
    if mean_obs == 0 or std_obs == 0 or std_sim == 0:
        return np.nan
    correlation = np.corrcoef(observed, simulated)[0, 1]
    return float(
        1.0
        - np.linalg.norm(
            (correlation - 1, std_sim / std_obs - 1, simulated.mean() / mean_obs - 1)
        )
    )


def _percent_ratio(numerator, denominator) -> float:
    return float(100.0 * numerator / denominator) if denominator != 0 else np.nan


def _fdc_biases(observed, simulated) -> tuple[float, float]:
    """Signed biases of independently ranked high- and low-flow tails."""
    observed, simulated = np.sort(observed)[::-1], np.sort(simulated)[::-1]
    n_high = max(int(np.round(0.02 * observed.size)), 1)
    n_low = max(int(np.round(0.20 * observed.size)), 1)
    high_obs, high_sim = observed[:n_high], simulated[:n_high]
    fhv = _percent_ratio((high_sim - high_obs).sum(), high_obs.sum())
    low_obs, low_sim = observed[-n_low:], simulated[-n_low:]
    log_obs = np.log(np.where(low_obs <= 0, 1e-6, low_obs))
    log_sim = np.log(np.where(low_sim <= 0, 1e-6, low_sim))
    obs_sum = (log_obs - log_obs.min()).sum()
    sim_sum = (log_sim - log_sim.min()).sum()
    return fhv, _percent_ratio(obs_sum - sim_sum, obs_sum)


def validation_nse(results: dict[str, pd.DataFrame]) -> tuple[float, int]:
    values = np.array([_nse(*_paired_values(frame)) for frame in results.values()])
    valid = values[np.isfinite(values)]
    if not valid.size:
        raise ValueError(
            "Validation NSE is undefined for every basin. Check observed runoff and date coverage."
        )
    return float(np.median(valid)), int(valid.size)


def evaluate(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = ["NSE", "KGE", "RMSE", "PBIAS", "FHV_2", "FLV_20"]
    records = {}
    for basin, frame in results.items():
        observed, simulated = _paired_values(frame)
        if not observed.size:
            records[basin] = [np.nan] * len(names)
            continue
        error = simulated - observed
        records[basin] = [
            _nse(observed, simulated),
            _kge(observed, simulated),
            np.sqrt(np.mean(error**2)),
            _percent_ratio(error.sum(), observed.sum()),
            *_fdc_biases(observed, simulated),
        ]
    return pd.DataFrame.from_dict(records, orient="index", columns=names).rename_axis(
        "basin_id"
    )
