"""TriFlow-MoE settings for the CAMELS-US temporal experiment."""

PERIODS = {
    "train": ("1999-10-01", "2008-09-30"),
    "validation": ("1980-10-01", "1989-09-30"),
    "test": ("1989-10-01", "1999-09-30"),
}
SEQUENCE_LENGTH = 365
WARMUP_DAYS = 365

MODEL_PARAMETERS = {
    "d_model": 256,
    "d_k": 128,
    "d_hidden_gate": 32,
    "drop_out_rate": 0.1,
    "quick_window": 7,
    "inter_window": 30,
    "static_hidden_size": 64,
    "static_context_size": 64,
    "static_dropout": 0.1,
    "use_doy_feature": True,
    "quick_state_dependent": True,
    "quick_state_weight_max": 4.0,
    "quick_state_weight_init": 1.0,
    "quick_storage_scale_bounds": [1.0, 100.0],
    "quick_meteo_logit_limit": 0.5,
    "ledger_kappa_bounds": [[0.25, 0.95], [0.03, 0.25], [0.0005, 0.03]],
    "ledger_soil_capacity_bounds": [25.0, 600.0],
    "ledger_pet_scale_bounds": [0.5, 1.5],
    "ledger_et_threshold_bounds": [0.3, 1.0],
    "ledger_soil_init_fraction": 0.5,
    "ledger_cap_tau": 1.0,
    "ledger_float64": False,
    "use_snow_module": True,
    "snow_transition_width": 1.0,
    "snow_initial_ddf": 3.0,
    "snow_n_bands": 5,
    "snow_lapse_rate": 6.5,
    "snow_seasonal_melt": True,
    "snow_initial_ddf_ratio": 0.5,
    "snow_cold_content": True,
    "snow_ripeness_scale": 1.0,
    "snow_detach_features": True,
    "fast_scans": True,
}

BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
LR_STEP_EPOCHS = 5
LR_GAMMA = 0.8
PATIENCE = 20
GRAD_CLIP_NORM = 1.0
USAGE_FLOOR = 0.05
USAGE_WEIGHT = 0.01
DEFAULT_SEED = 325
