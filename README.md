# TriFlow-MoE

This repository contains the TriFlow-MoE implementation accompanying the manuscript submitted to *Water Resources Research*. The model simulates daily basin runoff with a water-conserving snow and soil backbone, a recharge allocation network, and three causal attention experts for fast, intermediate, and slow reservoir responses.

## CAMELS-US data

The experiment uses the 531-basin CAMELS-US benchmark subset described by [Newman et al. (2017)](https://doi.org/10.1175/JHM-D-16-0284.1). The exact USGS gauge IDs are included in `data/basins_camels_us_531.txt`. The list was taken from the [NeuralHydrology benchmark example](https://github.com/neuralhydrology/neuralhydrology/blob/2dc8eb218ae9561b45ec59b4471d5eace96305d8/examples/06-Finetuning/531_basin_list.txt) and checked against the [Kratzert et al. (2019) replication repository](https://github.com/kratzert/lstm_for_pub/blob/36366e2cd32756fb07abc533ecbb7672a4738bc6/data/basin_list.txt).

Download the following CAMELS-US inputs from the [NSF NCAR archive](https://doi.org/10.5065/D6MW2F4D) or its [Zenodo mirror](https://zenodo.org/records/15529996).

| Input | Files | Dataset citation |
| --- | --- | --- |
| Daily Daymet basin-mean forcings and USGS runoff observations | `basin_timeseries_v1p2_metForcing_obsFlow.zip` | [Newman et al. (2015)](https://doi.org/10.5194/hess-19-209-2015) |
| Static basin attributes and gauge latitude | `camels_topo.txt`, `camels_vege.txt`, `camels_soil.txt`, `camels_geol.txt` | [Addor et al. (2017)](https://doi.org/10.5194/hess-21-5293-2017), [attribute data DOI](https://doi.org/10.5065/D6G73C3Q) |

Extract the time series and place the attribute text files in the following layout. The loader also accepts `basin_mean_forcing/` and `usgs_streamflow/` directly under the data directory.

```text
CAMELS_US/
├── basin_dataset_public_v1p2/
│   ├── basin_mean_forcing/daymet/<HUC>/*_forcing_leap.txt
│   └── usgs_streamflow/<HUC>/*_streamflow_qc.txt
└── camels_attributes_v2.0/
    ├── camels_topo.txt
    ├── camels_vege.txt
    ├── camels_soil.txt
    └── camels_geol.txt
```

Existing prepared forcings in `basin_mean_forcing/all/<HUC>/<gauge_id>.csv` are also supported. These files have the basin area in m² on the first line and the Daymet columns on the CSV header line. Automatic detection prefers that layout when `all/` exists. Use `--forcing-format daymet` to select the original Daymet text files explicitly.

The neural inputs are daily precipitation, shortwave radiation, maximum and minimum air temperature, and vapor pressure, together with 13 static attributes. The physical backbone receives unstandardized forcings and [Hargreaves–Samani PET](https://doi.org/10.13031/2013.26773). Observations and predictions are expressed in mm/day.

## Run the experiment

Use Python 3.10 or later. Install the dependencies from the repository root. For CUDA, use the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to select the build for your system.

```bash
python -m pip install -r requirements.txt
python code/run_CAMELS.py --help
python -u code/run_CAMELS.py --data-dir /path/to/CAMELS_US --device cuda:0 --seed 325
```

Each invocation runs one seed. Use `--seed` for the other independent runs. Device `auto` selects CUDA device 0 if available and otherwise uses the CPU. `--device cpu`, `--debug-basins N`, and `--epochs 1` are available for local checks. `--batch-size` and `--eval-batch-size` can be lowered for devices with less memory.

| Split | First target day | Last target day |
| --- | --- | --- |
| Training | 1999-10-01 | 2008-09-30 |
| Validation | 1980-10-01 | 1989-09-30 |
| Testing | 1989-10-01 | 1999-09-30 |

All splits use the same gauge list. Each prediction uses a trailing 365-day input window and up to 365 preceding days of forcing for physical state initialization. Training-date forcings and observations determine the dynamic standardization and runoff-loss weights, respectively. The model is selected by median validation NSE. Adam starts at a learning rate of 0.001, decayed by 0.8 every five epochs. Training runs for up to 100 epochs with an early-stopping patience of 20 epochs.

### Selected hyperparameters in Table S9

| Hyperparameter | Default |
| --- | --- |
| Embedding dimension | 256 |
| Attention query and key dimension | 128 |
| Gating network hidden dimension | 32 |
| Dropout rate | 0.1 |
| Fast expert attention window | 7 days |
| Intermediate expert attention window | 30 days |
| Slow expert attention history | All available causal history within the 365-day input window |
| Static-attribute hidden dimension | 64 |
| Training batch size | 256 |
| Initial learning rate | 0.001 |

The default settings are in `code/triflow_moe/config.py`. Each run creates a separate directory under `results/` or `--output-dir`. It contains `best_model.pt`, the run and model configurations, the training scaler, `test_predictions.csv.gz`, and basin-level and summary tables for NSE, KGE, RMSE, PBIAS, FHV₂, and FLV₂₀. The test predictions CSV has columns `basin_id`, `date`, `y_obs_mm_d`, and `y_sim_mm_d`.

## Code acknowledgments

We thank the following authors for releasing the software that informed or underlies the CAMELS-US workflow. The training and data-handling workflow in the original implementation was adapted from Heudorfer et al., whose code credits Hy2DL and NeuralHydrology. Upstream license notices are included in `licenses/`.

- Heudorfer, B., Gupta, H. V., & Loritz, R. (2025). Are Deep Learning Models in Hydrology Entity Aware? *Geophysical Research Letters*, 52(6), e2024GL113036. [Article](https://doi.org/10.1029/2024GL113036) · [Source code](https://github.com/bheudorfer/2025_entity_awareness_hydrology_camels_us) · [Archived code and data, v2](https://doi.org/10.5281/zenodo.14871156).
- Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., & Ehret, U. (2024). To bucket or not to bucket? Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization. *Hydrology and Earth System Sciences*, 28, 2705–2719. [Article](https://doi.org/10.5194/hess-28-2705-2024) · [Hy2DL source code, v1.0](https://github.com/KIT-HYD/Hy2DL/tree/v1.0) · [Archived code and data](https://doi.org/10.5281/zenodo.11103634).
- Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for Deep Learning research in hydrology. *Journal of Open Source Software*, 7(71), 4050. [Article](https://doi.org/10.21105/joss.04050) · [Source code](https://github.com/neuralhydrology/neuralhydrology).

The basin-scaled NSE objective follows [Kratzert et al. (2019)](https://doi.org/10.5194/hess-23-5089-2019).
