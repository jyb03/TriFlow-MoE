"""CAMELS-US loading, training statistics, and causal daily windows."""

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DAYMET_COLUMNS = {
    "prcp(mm/day)": "prcp_daymet",
    "srad(W/m2)": "srad_daymet",
    "tmax(C)": "tmax_daymet",
    "tmin(C)": "tmin_daymet",
    "vp(Pa)": "vp_daymet",
}
DYNAMIC_COLUMNS = tuple(DAYMET_COLUMNS.values())
STATIC_COLUMNS = (
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_diff",
    "gvf_diff",
    "soil_depth_pelletier",
    "soil_porosity",
    "soil_conductivity",
    "sand_frac",
    "clay_frac",
    "carbonate_rocks_frac",
    "geol_permeability",
)
RAW_COLUMNS = ("prcp_daymet", "tmax_daymet", "tmin_daymet", "PET_hargreaves(mm/d)")
TARGET = "QObs(mm/d)"
CFS_TO_MM_PER_DAY_TIMES_AREA = 0.028316846592 * 86_400 * 1_000


def read_basin_ids(path: Path) -> list[str]:
    """Read USGS gauge IDs while preserving their leading zeros and order."""
    ids = path.read_text(encoding="utf-8-sig").split()
    if not ids or any(re.fullmatch(r"[0-9]{8}", basin) is None for basin in ids):
        raise ValueError(f"Expected a list of eight-digit USGS gauge IDs in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate USGS gauge IDs in {path}")
    return ids


def hargreaves_pet(tmax, tmin, doy, latitude):
    """Hargreaves–Samani PET in mm/day with FAO-56 extraterrestrial radiation."""
    phi = np.deg2rad(latitude)
    angle = 2.0 * np.pi * doy / 365.0
    distance = 1.0 + 0.033 * np.cos(angle)
    declination = 0.409 * np.sin(angle - 1.39)
    sunset = np.arccos(np.clip(-np.tan(phi) * np.tan(declination), -1.0, 1.0))
    radiation = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * distance
        * (
            sunset * np.sin(phi) * np.sin(declination)
            + np.cos(phi) * np.cos(declination) * np.sin(sunset)
        )
    )
    tmean = 0.5 * (tmax + tmin)
    temperature_range = np.sqrt(np.maximum(tmax - tmin, 0.0))
    return np.maximum(
        0.0023 * (0.408 * radiation) * (tmean + 17.8) * temperature_range, 0.0
    )


def _date_index(frame: pd.DataFrame, path: Path) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": frame["Year"],
                "month": frame["Mnth"],
                "day": frame["Day"],
            }
        )
    )
    if dates.empty or dates.hasnans or not dates.is_unique:
        raise ValueError(f"Expected nonempty, unique daily dates in {path}")
    return dates


def _basin_paths(root: Path, pattern: str, basin_ids: Sequence[str]) -> dict[str, Path]:
    selected = set(basin_ids)
    result = {}
    for path in root.rglob(pattern):
        basin = path.stem.split("_", 1)[0]
        if basin in selected:
            if basin in result:
                raise ValueError(f"Multiple files for basin {basin} in {root}")
            result[basin] = path
    missing = [basin for basin in basin_ids if basin not in result]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} basin file(s) matching {pattern} in {root}. "
            f"First missing IDs: {', '.join(missing[:8])}"
        )
    return result


class CamelsUS:
    """Read the public Daymet text layout or the prepared CAMELS CSV layout."""

    def __init__(self, root: Path, basin_ids: Sequence[str], forcing_format: str):
        timeseries = root / "basin_timeseries_v1p2_metForcing_obsFlow"
        candidates = (
            root,
            root / "basin_dataset_public_v1p2",
            timeseries,
            timeseries / "basin_dataset_public_v1p2",
        )
        series_root = next(
            (
                p
                for p in candidates
                if (p / "basin_mean_forcing").is_dir()
                and (p / "usgs_streamflow").is_dir()
            ),
            None,
        )
        if series_root is None:
            raise FileNotFoundError(
                f"CAMELS-US forcing and USGS streamflow folders are missing in {root}"
            )
        attributes = next(
            (
                p / "camels_attributes_v2.0"
                for p in (root, series_root, series_root.parent)
                if (p / "camels_attributes_v2.0").is_dir()
            ),
            None,
        )
        if attributes is None:
            raise FileNotFoundError(
                f"Put the CAMELS-US attribute tables in {root / 'camels_attributes_v2.0'}"
            )

        forcing_root = series_root / "basin_mean_forcing"
        legacy = forcing_root / "all"
        self.forcing_format = forcing_format
        if self.forcing_format == "auto":
            self.forcing_format = "legacy" if legacy.is_dir() else "daymet"
        if self.forcing_format == "legacy":
            self.forcing_paths = _basin_paths(legacy, "*.csv", basin_ids)
        else:
            daymet = forcing_root / "daymet"
            if not daymet.is_dir() and (forcing_root / "Daymet").is_dir():
                daymet = forcing_root / "Daymet"
            self.forcing_paths = _basin_paths(daymet, "*_forcing_leap.txt", basin_ids)
        self.discharge_paths = _basin_paths(
            series_root / "usgs_streamflow", "*_streamflow_qc.txt", basin_ids
        )
        self.basin_ids = list(basin_ids)
        tables = []
        for name in ("topo", "vege", "soil", "geol"):
            path = attributes / f"camels_{name}.txt"
            table = pd.read_csv(path, sep=";", dtype={"gauge_id": str}).set_index(
                "gauge_id"
            )
            if not table.index.is_unique:
                raise ValueError(f"Duplicate gauge IDs in {path}")
            tables.append(table)
        fields = [*STATIC_COLUMNS, "gauge_lat"]
        selected = pd.concat(tables, axis=1).reindex(index=basin_ids, columns=fields)
        selected = selected.apply(pd.to_numeric, errors="coerce")
        invalid = ~np.isfinite(selected.to_numpy())
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise ValueError(
                f"Missing or invalid attribute {fields[column]} for basin {basin_ids[row]}"
            )
        self.attributes = selected.loc[:, list(STATIC_COLUMNS)]
        self.latitudes = selected["gauge_lat"]

    def read_basin(self, basin: str) -> pd.DataFrame:
        path = self.forcing_paths[basin]
        with path.open(encoding="utf-8-sig") as stream:
            if self.forcing_format == "daymet":
                stream.readline()  # Gauge latitude in the forcing header
                stream.readline()  # Basin mean elevation in the forcing header
            area = float(stream.readline().strip())
            if not np.isfinite(area) or area <= 0:
                raise ValueError(f"Invalid basin area in {path}")
            if self.forcing_format == "daymet":
                forcing = pd.read_csv(stream, sep=r"\s+").rename(columns=DAYMET_COLUMNS)
            else:
                forcing = pd.read_csv(stream)

        forcing.index = _date_index(forcing, path)
        missing = [column for column in DYNAMIC_COLUMNS if column not in forcing]
        if missing:
            raise ValueError(f"Missing Daymet columns {missing} in {path}")
        forcing = forcing.loc[:, list(DYNAMIC_COLUMNS)].astype(float).sort_index()
        forcing = forcing.replace([np.inf, -np.inf, -9999.0], np.nan)
        forcing["prcp_daymet"] = forcing["prcp_daymet"].where(
            forcing["prcp_daymet"] >= 0
        )
        dates = pd.date_range(forcing.index[0], forcing.index[-1], freq="D")
        frame = forcing.reindex(dates)
        frame[RAW_COLUMNS[-1]] = hargreaves_pet(
            frame["tmax_daymet"].to_numpy(),
            frame["tmin_daymet"].to_numpy(),
            dates.dayofyear.to_numpy(),
            self.latitudes[basin],
        )

        path = self.discharge_paths[basin]
        discharge = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["basin", "Year", "Mnth", "Day", "cfs", "flag"],
        )
        discharge.index = _date_index(discharge, path)
        cfs = discharge["cfs"].astype(float)
        cfs = cfs.where(np.isfinite(cfs) & (cfs >= 0))
        frame[TARGET] = cfs * (CFS_TO_MM_PER_DAY_TIMES_AREA / area)
        return frame


def _unit_scale(std: np.ndarray) -> np.ndarray:
    """Use scale 1 for constant columns, including single-basin debug runs."""
    return np.where(np.isfinite(std) & (std > 0), std, 1.0)


def fit_scaler(
    frames: dict[str, pd.DataFrame],
    attributes: pd.DataFrame,
    training_period: tuple[str, str],
) -> dict:
    """Fit meteorological and runoff statistics using only training dates."""
    start, end = training_period
    dynamic_parts = []
    basin_std = {}
    for basin, frame in frames.items():
        training = frame.loc[start:end]
        observed = training[TARGET].dropna().to_numpy()
        if not observed.size:
            raise ValueError(
                f"No observed training runoff for basin {basin} between {start} and {end}"
            )
        basin_std[basin] = float(np.std(observed))
        dynamic_parts.append(training.loc[:, list(DYNAMIC_COLUMNS)].to_numpy())

    dynamic = np.concatenate(dynamic_parts)
    if not np.isfinite(dynamic).any(axis=0).all():
        raise ValueError(
            "At least one Daymet input has no valid values in the training period"
        )
    return {
        "x_d_mean": np.nanmean(dynamic, axis=0).astype(np.float32).tolist(),
        "x_d_std": _unit_scale(np.nanstd(dynamic, axis=0)).astype(np.float32).tolist(),
        "x_s_mean": attributes.mean().to_numpy(dtype=np.float32).tolist(),
        "x_s_std": _unit_scale(attributes.std().to_numpy()).astype(np.float32).tolist(),
        "basin_std_mm_d": basin_std,
    }


class BasinWindows(Dataset):
    """A split-specific view of valid target days for one basin."""

    def __init__(self, series, period: tuple[str, str], require_target: bool):
        self.series = series
        start = series.dates.searchsorted(pd.Timestamp(period[0]), side="left")
        end = series.dates.searchsorted(pd.Timestamp(period[1]), side="right")
        targets = np.arange(max(start, series.sequence_length - 1), end)
        current_start = targets - series.sequence_length + 1
        history_start = np.maximum(0, current_start - series.warmup_days)
        # Prefix counts reject any missing forcing within either input segment.
        valid = series.bad_dynamic[targets + 1] == series.bad_dynamic[current_start]
        valid &= series.bad_raw[targets + 1] == series.bad_raw[history_start]
        if require_target:
            valid &= np.isfinite(series.target[targets].numpy())
        self.indices = targets[valid]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        series = self.series
        target = int(self.indices[index])
        start = target - series.sequence_length + 1
        padded_start = start + series.warmup_days
        return {
            "x_dynamic": series.dynamic[start : target + 1],
            "x_static": series.static,
            "x_raw": series.raw[padded_start : target + series.warmup_days + 1],
            "x_raw_warmup": series.raw[start:padded_start],
            "raw_warmup_mask": series.history_mask[start:padded_start],
            "doy": series.doy[start : target + 1],
            "y_obs": series.target[target : target + 1],
            "basin_std": series.basin_std,
        }

    @property
    def dates(self):
        return self.series.dates[self.indices]


class BasinSeries:
    """Store each basin's inputs once for all three temporal splits."""

    def __init__(
        self,
        basin: str,
        frame: pd.DataFrame,
        attributes: pd.Series,
        scaler: dict,
        sequence_length: int,
        warmup_days: int,
    ):
        self.dates = frame.index
        self.doy = torch.tensor(frame.index.dayofyear.to_numpy(), dtype=torch.float32)
        self.sequence_length = sequence_length
        self.warmup_days = warmup_days
        dynamic = frame.loc[:, list(DYNAMIC_COLUMNS)].to_numpy(dtype=np.float32)
        raw = frame.loc[:, list(RAW_COLUMNS)].to_numpy(dtype=np.float32)
        self.bad_dynamic = np.r_[0, (~np.isfinite(dynamic).all(axis=1)).cumsum()]
        self.bad_raw = np.r_[0, (~np.isfinite(raw).all(axis=1)).cumsum()]
        self.dynamic = (
            torch.from_numpy(dynamic) - torch.tensor(scaler["x_d_mean"])
        ) / torch.tensor(scaler["x_d_std"])
        self.static = (
            torch.tensor(attributes.to_numpy(dtype=np.float32))
            - torch.tensor(scaler["x_s_mean"])
        ) / torch.tensor(scaler["x_s_std"])
        self.target = torch.tensor(frame[TARGET].to_numpy(), dtype=torch.float32)
        self.basin_std = torch.tensor(
            [scaler["basin_std_mm_d"][basin]], dtype=torch.float32
        )
        # Mask padding only before the forcing record. Gaps inside it invalidate windows.
        self.raw = torch.cat(
            (torch.zeros(warmup_days, raw.shape[1]), torch.from_numpy(raw))
        )
        self.history_mask = torch.arange(len(self.raw)) >= warmup_days


def build_datasets(
    archive: CamelsUS,
    periods: dict[str, tuple[str, str]],
    sequence_length: int,
    warmup_days: int,
) -> tuple[dict, dict]:
    frames = {basin: archive.read_basin(basin) for basin in archive.basin_ids}
    scaler = fit_scaler(frames, archive.attributes, periods["train"])
    splits = {split: {} for split in periods}
    for basin in archive.basin_ids:
        series = BasinSeries(
            basin,
            frames.pop(basin),
            archive.attributes.loc[basin],
            scaler,
            sequence_length,
            warmup_days,
        )
        for split, period in periods.items():
            windows = BasinWindows(series, period, require_target=(split == "train"))
            if not len(windows):
                raise ValueError(
                    f"No valid {split} windows for basin {basin} from {period[0]} to {period[1]}"
                )
            splits[split][basin] = windows
    return splits, scaler
