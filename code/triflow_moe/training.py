"""TriFlow-MoE batch inputs, training objective, and checkpoint selection."""

import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import config
from .metrics import validation_nse

LOGGER = logging.getLogger(__name__)


def forward_batch(model, sample, device):
    dynamic = sample["x_dynamic"].to(device)
    raw = sample["x_raw"].to(device)
    history = sample["x_raw_warmup"].to(device)
    return model(
        dynamic,
        x_static=sample["x_static"].to(device),
        doy=sample["doy"].to(device),
        p_raw=raw[..., 0],
        t_raw=0.5 * (raw[..., 1] + raw[..., 2]),
        pet_raw=raw[..., 3],
        warmup_p=history[..., 0],
        warmup_t=0.5 * (history[..., 1] + history[..., 2]),
        warmup_pet=history[..., 3],
        warmup_mask=sample["raw_warmup_mask"].to(device),
    )


def training_loss(prediction, target, basin_std):
    """Basin-scaled squared error with a recharge allocation floor."""
    squared_error = (prediction["y_hat"] - target).square()
    weights = 1 / (basin_std + 0.1).square()
    runoff_loss = (weights * squared_error).mean()
    mean_partition = prediction["partition"].mean(dim=(0, 1))
    usage_loss = (
        (config.USAGE_FLOOR - mean_partition).clamp_min(0) / (config.USAGE_FLOOR + 1e-6)
    ).mean()
    return runoff_loss + config.USAGE_WEIGHT * usage_loss


@torch.inference_mode()
def predict(model, basins, batch_size: int, device) -> dict[str, pd.DataFrame]:
    model.eval()
    results = {}
    for basin, windows in basins.items():
        observed, simulated = [], []
        for batch in DataLoader(windows, batch_size=batch_size):
            output = forward_batch(model, batch, device)["y_hat"]
            observed.append(batch["y_obs"].numpy().reshape(-1))
            simulated.append(output.cpu().numpy().reshape(-1))
        frame = pd.DataFrame(
            {"y_obs": np.concatenate(observed), "y_sim": np.concatenate(simulated)},
            index=windows.dates,
        )
        if not np.isfinite(frame["y_sim"]).all():
            raise RuntimeError(f"Non-finite runoff prediction for basin {basin}")
        results[basin] = frame
    return results


def train(
    model,
    splits,
    *,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    device,
    checkpoint: Path,
) -> tuple[int, float]:
    training = ConcatDataset(list(splits["train"].values()))
    loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(training) >= batch_size),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.LR_STEP_EPOCHS, gamma=config.LR_GAMMA
    )
    best_nse = -np.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_batch(model, batch, device)
            loss = training_loss(
                prediction, batch["y_obs"].to(device), batch["basin_std"].to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRAD_CLIP_NORM, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(loss.item())
            del prediction, loss

        validation = predict(model, splits["validation"], eval_batch_size, device)
        median_nse, valid_count = validation_nse(validation)
        if median_nse > best_nse:
            best_nse = median_nse
            best_epoch = epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint)
        LOGGER.info(
            "Epoch %d | loss %.6g | validation median NSE %.5f (%d basins) | LR %.6g | %.1f s",
            epoch,
            np.mean(losses),
            median_nse,
            valid_count,
            scheduler.get_last_lr()[0],
            perf_counter() - start,
        )
        scheduler.step()
        if epoch - best_epoch >= config.PATIENCE:
            LOGGER.info(
                "Early stopping after %d epochs without improvement.", config.PATIENCE
            )
            break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return best_epoch, best_nse
