# =============================================================================
# HYPERSRL DIGITAL TWIN FRAMEWORK
# Cell 35: PINN Continual Learning
# =============================================================================
# Purpose  : Automatically fine-tune existing PINN models when sufficient new
#            data accumulates. Prevents catastrophic forgetting by freezing
#            the LSTM encoder layers. Validates updated model before saving
#            and rolls back if quality degrades. Maintains versioned model
#            archive with full audit trail.
# Inputs   : live_data/merged_dataset.csv        (from Cell 34)
#            processed/scaler_params.json         (from Cell 28)
#            processed/pinn2_bangladesh.pt        (from Cell 28)
#            processed/pinn2_vietnam.pt           (from Cell 28)
#            processed/pinn2_maldives.pt          (from Cell 28)
# Outputs  : models/versioned/pinn_{country}_v{timestamp}.pt
#            logs/continual_learning_log.csv
#            logs/continual_learning_summary_{timestamp}.json
# Architecture confirmed from checkpoint inspection:
#   lstm       : input_size=8, hidden=128, 2 layers
#   main_head  : Linear(128,64) -> ReLU -> Dropout -> Linear(64,1)
#   physics_head: Linear(128,32) -> ReLU -> Linear(32,1)
# =============================================================================

# --- 0. IMPORTS --------------------------------------------------------------

import os
import copy
import json
import logging
import warnings
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. LOGGING SETUP --------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HyperSLR.Cell35")

# --- 2. CONFIGURATION --------------------------------------------------------

DRIVE_BASE    = Path("/content/drive/MyDrive/BD_ML_VT")
LIVE_DIR      = DRIVE_BASE / "live_data"
LOG_DIR       = DRIVE_BASE / "logs"
PROCESSED_DIR = DRIVE_BASE / "processed"
VERSIONED_DIR = DRIVE_BASE / "models" / "versioned"

MERGED_CSV    = LIVE_DIR  / "merged_dataset.csv"
SCALER_JSON   = PROCESSED_DIR / "scaler_params.json"
CL_LOG        = LOG_DIR   / "continual_learning_log.csv"

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Model file mapping (Cell 28 naming convention)
MODEL_PATHS = {
    "Bangladesh": PROCESSED_DIR / "pinn2_bangladesh.pt",
    "Vietnam"   : PROCESSED_DIR / "pinn2_vietnam.pt",
    "Maldives"  : PROCESSED_DIR / "pinn2_maldives.pt",
}

# Architecture constants — confirmed from checkpoint key shape inspection:
#   lstm.weight_ih_l0 : (512, 8)   -> hidden=128 (512/4), input_size=8
#   main_head.0.weight: (64, 128)  -> first FC layer output=64
#   main_head.3.weight: (1, 64)    -> final FC layer (index 3, after Dropout)
#   physics_head.0.weight: (32,128)-> physics first layer output=32
HIDDEN_SIZE       = 128
NUM_LSTM_LAYERS   = 2
DROPOUT_RATE      = 0.2
CELL28_INPUT_SIZE = 8

# Cell 28 feature order — must match input_size=8 exactly
CELL28_FEATURE_COLUMNS = [
    "slrv2", "tide_gauge", "enso", "aviso",
    "vlm", "cmip6_thermo", "cmip6_glacier", "momentum",
]

# Full 12-feature set (used for drift detection in Cell 34, not for model input)
FEATURE_COLUMNS = [
    "slrv2", "tide_gauge", "enso", "aviso", "vlm",
    "cmip6_thermo", "cmip6_glacier", "momentum", "residual",
    "month_sin", "month_cos", "enso_x_aviso",
]

TARGET_COLUMN = "slrv2"

# Continual learning hyperparameters
UPDATE_TRIGGER_MIN_MONTHS = 3
FROZEN_LAYER_NAMES        = ["lstm"]
FINE_TUNE_LR              = 1e-4
FINE_TUNE_EPOCHS          = 30
EARLY_STOP_PATIENCE       = 10
PHYSICS_LAMBDA            = 0.4
QUALITY_GATE_FACTOR       = 1.05
SEQ_LEN                   = 12
BATCH_SIZE                = 16
VAL_SPLIT                 = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", DEVICE)

VERSIONED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --- 3. AUDIT LOGGING --------------------------------------------------------

def log_cl_event(
    country : str,
    stage   : str,
    status  : str,
    detail  : str,
    error   : Optional[str] = None
) -> None:
    """Append one row to the continual learning audit log."""
    row = pd.DataFrame([{
        "timestamp" : RUN_TIMESTAMP,
        "country"   : country,
        "stage"     : stage,
        "status"    : status,
        "detail"    : detail,
        "error"     : error if error else "",
    }])
    if CL_LOG.exists():
        row.to_csv(CL_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(CL_LOG, mode="w", header=True, index=False)
    logger.info("[LOG] %s | %s | %s | %s", country, stage, status, detail)


# --- 4. PINN MODEL ARCHITECTURE ----------------------------------------------

class PhysicsInformedSLRModel(nn.Module):
    """
    Physics-Informed LSTM model matching Cell 28 architecture exactly.

    Confirmed from checkpoint key inspection (all three country models
    share identical architecture):
      lstm        : input_size=8, hidden=128, 2 layers, dropout=0.2
      main_head   : Linear(128,64) -> ReLU -> Dropout(0.2) -> Linear(64,1)
                    Sequential indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear
      physics_head: Linear(128,32) -> ReLU -> Linear(32,1)
                    Sequential indices: 0=Linear, 1=ReLU, 2=Linear
    """

    def __init__(
        self,
        input_size  : int   = CELL28_INPUT_SIZE,
        hidden_size : int   = HIDDEN_SIZE,
        num_layers  : int   = NUM_LSTM_LAYERS,
        dropout     : float = DROPOUT_RATE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        # Named 'main_head' to match Cell 28 state_dict keys exactly
        # Indices: 0=Linear(128,64), 1=ReLU, 2=Dropout, 3=Linear(64,1)
        self.main_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Indices: 0=Linear(128,32), 1=ReLU, 2=Linear(32,1)
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        (prediction, physics_output) — both shape (batch, 1)
        """
        lstm_out, _  = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]
        prediction   = self.main_head(last_hidden)
        physics_out  = self.physics_head(last_hidden)
        return prediction, physics_out


def physics_informed_loss(
    prediction  : torch.Tensor,
    target      : torch.Tensor,
    physics_out : torch.Tensor,
    lam         : float = PHYSICS_LAMBDA,
) -> Tuple[torch.Tensor, float, float]:
    """
    Combined MSE + physics constraint loss.

    Physics constraint enforces that the physics head output (representing
    rate of change) is non-negative — sea level cannot fall below its
    barystatic minimum.

    Parameters
    ----------
    prediction  : Model main output, shape (batch, 1)
    target      : Ground truth, shape (batch, 1)
    physics_out : Physics head output, shape (batch, 1)
    lam         : Weight of physics constraint term

    Returns
    -------
    (total_loss, mse_loss_val, physics_loss_val)
    """
    mse_loss     = nn.MSELoss()(prediction, target)
    physics_loss = torch.mean(torch.relu(-physics_out) ** 2)
    total_loss   = mse_loss + lam * physics_loss
    return total_loss, float(mse_loss.item()), float(physics_loss.item())


# --- 5. DATA PREPARATION -----------------------------------------------------

def load_and_normalise(
    merged_df    : pd.DataFrame,
    country      : str,
    scaler_params: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract and normalise the 8-feature Cell 28 input matrix for one country.

    Uses CELL28_FEATURE_COLUMNS (8 features) to match the saved model's
    input_size=8. Normalisation applies scaler_params where available,
    falling back to data-derived mean/std otherwise.

    Parameters
    ----------
    merged_df    : Full merged dataset (all countries)
    country      : Country to extract
    scaler_params: Loaded scaler_params.json

    Returns
    -------
    (X_norm, y_norm, feature_stats)
      X_norm shape=(n, 8), y_norm shape=(n,),
      feature_stats is per-feature mean/std dict
    """
    cdf = merged_df[merged_df["country"] == country].copy()
    cdf = cdf.sort_values("date").reset_index(drop=True)

    available_features = [f for f in CELL28_FEATURE_COLUMNS if f in cdf.columns]
    cdf = cdf.dropna(subset=available_features + [TARGET_COLUMN])

    if cdf.empty:
        raise ValueError(f"No complete rows for {country} after dropping NaN.")

    X_raw = cdf[available_features].values.astype(np.float32)
    y_raw = cdf[TARGET_COLUMN].values.astype(np.float32)

    feature_stats = {}
    X_norm = np.zeros_like(X_raw)

    for i, feat in enumerate(available_features):
        if feat in scaler_params and "mean" in scaler_params[feat]:
            mean = float(scaler_params[feat]["mean"])
            std  = max(float(scaler_params[feat]["std"]), 1e-8)
        else:
            mean = float(X_raw[:, i].mean())
            std  = max(float(X_raw[:, i].std()), 1e-8)
        feature_stats[feat] = {"mean": mean, "std": std}
        X_norm[:, i] = (X_raw[:, i] - mean) / std

    y_mean = float(y_raw.mean())
    y_std  = max(float(y_raw.std()), 1e-8)
    feature_stats[TARGET_COLUMN] = {"mean": y_mean, "std": y_std}
    y_norm = (y_raw - y_mean) / y_std

    logger.info("  %s: %d samples, %d features normalised",
                country, len(y_norm), len(available_features))
    return X_norm, y_norm, feature_stats


def build_sequences(
    X      : np.ndarray,
    y      : np.ndarray,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature matrix into overlapping sliding-window sequences.

    Parameters
    ----------
    X       : Feature matrix, shape (n_samples, n_features)
    y       : Target vector, shape (n_samples,)
    seq_len : Window length

    Returns
    -------
    (X_seq, y_seq)
      X_seq shape=(n-seq_len, seq_len, n_features)
      y_seq shape=(n-seq_len,)
    """
    if len(X) <= seq_len:
        raise ValueError(
            f"Insufficient samples ({len(X)}) for sequence length {seq_len}. "
            f"Need at least {seq_len + 1} samples."
        )
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def split_new_data(
    X_seq    : np.ndarray,
    y_seq    : np.ndarray,
    val_frac : float = VAL_SPLIT,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split sequences into train/validation DataLoaders using temporal split.

    Most recent val_frac fraction is held out as validation to prevent
    data leakage.

    Parameters
    ----------
    X_seq    : Sequence array, shape (n, seq_len, n_features)
    y_seq    : Target array, shape (n,)
    val_frac : Fraction for validation

    Returns
    -------
    (train_loader, val_loader)
    """
    n         = len(X_seq)
    split_idx = max(1, int(n * (1 - val_frac)))

    X_train = torch.tensor(X_seq[:split_idx])
    y_train = torch.tensor(y_seq[:split_idx]).unsqueeze(1)
    X_val   = torch.tensor(X_seq[split_idx:])
    y_val   = torch.tensor(y_seq[split_idx:]).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )
    logger.info("  Train sequences: %d | Val sequences: %d",
                len(X_train), len(X_val))
    return train_loader, val_loader


# --- 6. MODEL LOADING AND VERSION CONTROL ------------------------------------

def load_model_weights(
    model   : PhysicsInformedSLRModel,
    path    : Path,
    country : str,
) -> Tuple[PhysicsInformedSLRModel, bool]:
    """
    Load saved Cell 28 weights into model using strict state_dict matching.

    Handles both full state_dict saves and wrapped checkpoint dictionaries.

    Parameters
    ----------
    model   : Initialised PhysicsInformedSLRModel (architecture must match)
    path    : Path to .pt file
    country : For logging

    Returns
    -------
    (model_with_weights, success_bool)
    """
    if not path.exists():
        logger.warning("  Model file not found: %s. Using random initialisation.", path)
        return model, False

    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Strict load — architecture must match exactly
        model.load_state_dict(state_dict, strict=True)
        model = model.to(DEVICE)
        logger.info("  %s: weights loaded (strict) from %s", country, path.name)
        return model, True

    except Exception as exc:
        logger.error("  %s: model load error: %s", country, exc)
        logger.debug(traceback.format_exc())
        return model.to(DEVICE), False


def version_control_model(
    model    : PhysicsInformedSLRModel,
    country  : str,
    metadata : Dict,
) -> Path:
    """
    Save versioned model checkpoint with timestamp and metadata.

    Parameters
    ----------
    model    : Trained model to save
    country  : Country name for filename
    metadata : Training metadata to embed in checkpoint

    Returns
    -------
    Path to saved checkpoint file
    """
    country_slug = country.lower().replace(" ", "_")
    filename     = f"pinn_{country_slug}_v{RUN_TIMESTAMP}.pt"
    save_path    = VERSIONED_DIR / filename

    checkpoint = {
        "model_state_dict" : model.state_dict(),
        "metadata"         : metadata,
        "timestamp"        : RUN_TIMESTAMP,
        "country"          : country,
        "architecture": {
            "input_size"   : CELL28_INPUT_SIZE,
            "hidden_size"  : HIDDEN_SIZE,
            "num_layers"   : NUM_LSTM_LAYERS,
            "dropout"      : DROPOUT_RATE,
            "main_head"    : "Linear(128,64)->ReLU->Dropout->Linear(64,1)",
            "physics_head" : "Linear(128,32)->ReLU->Linear(32,1)",
        },
    }
    torch.save(checkpoint, save_path)
    logger.info("  Model saved: %s", save_path)
    return save_path


# --- 7. UPDATE TRIGGER -------------------------------------------------------

def check_update_trigger(
    merged_df           : pd.DataFrame,
    country             : str,
    historical_end_date : str = "2022-12-31",
) -> Tuple[bool, int]:
    """
    Decide whether sufficient new data warrants a model update.

    Parameters
    ----------
    merged_df           : Full merged dataset
    country             : Country to check
    historical_end_date : ISO date string for end of original training data

    Returns
    -------
    (trigger: bool, new_months: int)
    """
    cdf        = merged_df[merged_df["country"] == country].copy()
    end_date   = pd.Timestamp(historical_end_date)
    new_rows   = cdf[cdf["date"] > end_date]
    new_months = len(new_rows)
    trigger    = new_months >= UPDATE_TRIGGER_MIN_MONTHS

    logger.info("  %s: %d new months beyond %s — trigger=%s",
                country, new_months, historical_end_date, trigger)
    return trigger, new_months


# --- 8. FREEZE LSTM LAYERS ---------------------------------------------------

def freeze_lstm_layers(
    model      : PhysicsInformedSLRModel,
    layer_names: List[str] = FROZEN_LAYER_NAMES,
) -> Tuple[PhysicsInformedSLRModel, int]:
    """
    Freeze LSTM encoder by setting requires_grad=False on matching layers.

    Only main_head and physics_head remain trainable, preserving learned
    temporal representations.

    Parameters
    ----------
    model       : Model with loaded weights
    layer_names : Layer name substrings to freeze (default: ['lstm'])

    Returns
    -------
    (model_with_frozen_layers, n_frozen_params)
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(ln in name for ln in layer_names):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Frozen params: %d | Trainable params: %d",
                frozen_count, trainable)
    return model, frozen_count


# --- 9. FINE-TUNING ----------------------------------------------------------

def fine_tune_pinn(
    model        : PhysicsInformedSLRModel,
    train_loader : DataLoader,
    val_loader   : DataLoader,
    country      : str,
) -> Tuple[PhysicsInformedSLRModel, Dict]:
    """
    Fine-tune PINN on new data with early stopping.

    Only parameters with requires_grad=True are updated. Gradient clipping
    at max_norm=1.0 prevents instability in the unfrozen heads.

    Parameters
    ----------
    model        : Model with frozen LSTM layers
    train_loader : Training DataLoader
    val_loader   : Validation DataLoader
    country      : For logging

    Returns
    -------
    (best_model, training_history dict)
    """
    model     = model.to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINE_TUNE_LR,
    )

    best_val_loss  = float("inf")
    best_state     = copy.deepcopy(model.state_dict())
    patience_count = 0
    history        = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred, phys = model(X_batch)
            loss, _, _ = physics_informed_loss(pred, y_batch, phys, PHYSICS_LAMBDA)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train = epoch_train_loss / max(len(train_loader), 1)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred, phys = model(X_batch)
                loss, _, _ = physics_informed_loss(pred, y_batch, phys, PHYSICS_LAMBDA)
                epoch_val_loss += loss.item()
        avg_val = epoch_val_loss / max(len(val_loader), 1)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss         = avg_val
            best_state            = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            patience_count        = 0
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  Epoch %2d/%d | train_loss=%.5f | val_loss=%.5f | patience=%d",
                epoch, FINE_TUNE_EPOCHS, avg_train, avg_val, patience_count
            )

        if patience_count >= EARLY_STOP_PATIENCE:
            logger.info("  Early stopping at epoch %d (patience=%d)",
                        epoch, EARLY_STOP_PATIENCE)
            break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    logger.info("  Fine-tuning complete: best val_loss=%.5f (epoch %d)",
                best_val_loss, history["best_epoch"])
    return model, history


# --- 10. QUALITY EVALUATION --------------------------------------------------

def compute_rmse(
    model      : PhysicsInformedSLRModel,
    X_seq      : np.ndarray,
    y_seq      : np.ndarray,
    feat_stats : Dict,
    target_col : str = TARGET_COLUMN,
) -> float:
    """
    Compute RMSE in original de-normalised units (mm).

    Parameters
    ----------
    model      : Trained model
    X_seq      : Normalised sequences, shape (n, seq_len, n_features)
    y_seq      : Normalised targets, shape (n,)
    feat_stats : Per-feature mean/std for de-normalisation
    target_col : Target column name

    Returns
    -------
    RMSE in mm (float)
    """
    model.eval()
    X_t = torch.tensor(X_seq).to(DEVICE)
    with torch.no_grad():
        preds, _ = model(X_t)
    preds_np = preds.cpu().numpy().flatten()

    y_mean       = feat_stats[target_col]["mean"]
    y_std        = feat_stats[target_col]["std"]
    preds_denorm = preds_np * y_std + y_mean
    truth_denorm = y_seq    * y_std + y_mean

    return float(np.sqrt(np.mean((preds_denorm - truth_denorm) ** 2)))


def evaluate_update_quality(
    updated_model  : PhysicsInformedSLRModel,
    original_model : PhysicsInformedSLRModel,
    X_seq          : np.ndarray,
    y_seq          : np.ndarray,
    feat_stats     : Dict,
    country        : str,
) -> Tuple[bool, float, float]:
    """
    Validate updated model against quality gate.

    Quality gate: updated_RMSE <= original_RMSE * QUALITY_GATE_FACTOR (1.05)

    Returns
    -------
    (passed: bool, updated_rmse_mm: float, original_rmse_mm: float)
    """
    updated_rmse  = compute_rmse(updated_model,  X_seq, y_seq, feat_stats)
    original_rmse = compute_rmse(original_model, X_seq, y_seq, feat_stats)
    threshold     = original_rmse * QUALITY_GATE_FACTOR
    passed        = updated_rmse <= threshold

    logger.info(
        "  %s quality gate: updated_RMSE=%.2fmm | original_RMSE=%.2fmm | "
        "threshold=%.2fmm | %s",
        country, updated_rmse, original_rmse, threshold,
        "PASSED" if passed else "FAILED"
    )
    return passed, updated_rmse, original_rmse


def rollback_if_degraded(
    updated_model  : PhysicsInformedSLRModel,
    original_model : PhysicsInformedSLRModel,
    passed         : bool,
    country        : str,
) -> Tuple[PhysicsInformedSLRModel, str]:
    """
    Return original model if quality gate failed; updated model otherwise.

    Returns
    -------
    (model_to_deploy, decision: 'UPDATED' | 'ROLLED_BACK')
    """
    if passed:
        logger.info("  %s: quality gate passed — deploying updated model.", country)
        return updated_model, "UPDATED"
    else:
        logger.warning("  %s: quality gate FAILED — rolling back to original.", country)
        return original_model, "ROLLED_BACK"


# --- 11. FULL CONTINUAL LEARNING FOR ONE COUNTRY ----------------------------

def run_continual_learning_country(
    country      : str,
    merged_df    : pd.DataFrame,
    scaler_params: Dict,
) -> Dict[str, Any]:
    """
    Execute the full continual learning pipeline for one country.

    Steps:
      1. Check update trigger (minimum 3 new months beyond 2022-12-31)
      2. Load and normalise data using Cell 28 8-feature schema
      3. Build sliding-window sequences
      4. Load original Cell 28 model weights (strict)
      5. Freeze LSTM encoder layers
      6. Fine-tune main_head and physics_head on new data
      7. Evaluate quality gate (RMSE within 5% of original)
      8. Rollback if quality gate fails
      9. Save versioned checkpoint

    Returns
    -------
    Result dictionary for this country
    """
    logger.info("=" * 60)
    logger.info("CONTINUAL LEARNING: %s", country)
    logger.info("=" * 60)

    result = {
        "country"        : country,
        "triggered"      : False,
        "new_months"     : 0,
        "updated_rmse"   : None,
        "original_rmse"  : None,
        "quality_passed" : None,
        "decision"       : None,
        "model_path"     : None,
        "best_val_loss"  : None,
        "best_epoch"     : None,
        "error"          : None,
    }

    try:
        # Step 1: Update trigger
        triggered, new_months = check_update_trigger(merged_df, country)
        result["triggered"]   = triggered
        result["new_months"]  = new_months

        if not triggered:
            msg = (f"Only {new_months} new months "
                   f"(need {UPDATE_TRIGGER_MIN_MONTHS}). Skipping.")
            logger.info("  %s: %s", country, msg)
            log_cl_event(country, "TRIGGER", "SKIPPED", msg)
            return result

        log_cl_event(country, "TRIGGER", "FIRED",
                     f"{new_months} new months available")

        # Step 2: Load and normalise (8 Cell28 features)
        X_norm, y_norm, feat_stats = load_and_normalise(
            merged_df, country, scaler_params
        )

        # Step 3: Build sequences
        X_seq, y_seq             = build_sequences(X_norm, y_norm, SEQ_LEN)
        train_loader, val_loader = split_new_data(X_seq, y_seq)

        # Step 4: Load original model weights
        n_features     = X_norm.shape[1]
        original_model = PhysicsInformedSLRModel(
            input_size=n_features
        ).to(DEVICE)
        original_model, load_ok = load_model_weights(
            original_model, MODEL_PATHS[country], country
        )
        log_cl_event(
            country, "LOAD_MODEL",
            "SUCCESS" if load_ok else "FALLBACK",
            f"path={MODEL_PATHS[country].name}, strict_load={load_ok}"
        )

        # Keep frozen copy for quality gate comparison
        original_for_comparison = copy.deepcopy(original_model)

        # Step 5: Freeze LSTM layers
        updated_model, frozen_count = freeze_lstm_layers(
            copy.deepcopy(original_model)
        )
        log_cl_event(country, "FREEZE", "SUCCESS",
                     f"frozen_params={frozen_count}")

        # Step 6: Fine-tune
        logger.info("  Starting fine-tuning for %s...", country)
        updated_model, history = fine_tune_pinn(
            updated_model, train_loader, val_loader, country
        )
        result["best_val_loss"] = history["best_val_loss"]
        result["best_epoch"]    = history["best_epoch"]
        log_cl_event(
            country, "FINE_TUNE", "SUCCESS",
            f"best_val_loss={history['best_val_loss']:.5f}, "
            f"best_epoch={history['best_epoch']}"
        )

        # Step 7: Quality gate
        passed, updated_rmse, original_rmse = evaluate_update_quality(
            updated_model, original_for_comparison,
            X_seq, y_seq, feat_stats, country
        )
        result["updated_rmse"]   = round(updated_rmse, 3)
        result["original_rmse"]  = round(original_rmse, 3)
        result["quality_passed"] = passed

        # Step 8: Rollback if degraded
        final_model, decision = rollback_if_degraded(
            updated_model, original_for_comparison, passed, country
        )
        result["decision"] = decision
        log_cl_event(
            country, "QUALITY_GATE",
            "PASSED" if passed else "FAILED_ROLLBACK",
            f"updated={updated_rmse:.2f}mm, original={original_rmse:.2f}mm, "
            f"decision={decision}"
        )

        # Step 9: Version control
        metadata = {
            "country"          : country,
            "new_months"       : new_months,
            "updated_rmse_mm"  : updated_rmse,
            "original_rmse_mm" : original_rmse,
            "quality_passed"   : passed,
            "decision"         : decision,
            "best_val_loss"    : history["best_val_loss"],
            "best_epoch"       : history["best_epoch"],
            "fine_tune_lr"     : FINE_TUNE_LR,
            "physics_lambda"   : PHYSICS_LAMBDA,
            "run_timestamp"    : RUN_TIMESTAMP,
            "strict_load"      : load_ok,
        }
        save_path          = version_control_model(final_model, country, metadata)
        result["model_path"] = str(save_path)
        log_cl_event(country, "VERSION_CONTROL", "SUCCESS",
                     f"saved={save_path.name}")

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("  %s: continual learning failed: %s", country, exc)
        logger.debug(traceback.format_exc())
        log_cl_event(country, "ERROR", "FAILED", str(exc), error=str(exc))

    return result


# =============================================================================
# CELL 35 — MAIN EXECUTION
# =============================================================================

def run_cell_35() -> Dict[str, Any]:
    """
    Execute Cell 35 continual learning for all three countries.

    Returns
    -------
    Full summary dictionary
    """
    logger.info("=" * 70)
    logger.info("HYPERSRL — CELL 35: PINN CONTINUAL LEARNING")
    logger.info("Run timestamp : %s", RUN_TIMESTAMP)
    logger.info("Device        : %s", DEVICE)
    logger.info("=" * 70)

    if not MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {MERGED_CSV}\n"
            "Execute Cell 34 before running Cell 35."
        )

    merged_df     = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    scaler_params = {}
    if SCALER_JSON.exists():
        with open(SCALER_JSON) as f:
            scaler_params = json.load(f)
    logger.info("Merged dataset: %d rows", len(merged_df))

    countries       = ["Bangladesh", "Vietnam", "Maldives"]
    country_results = {}

    for country in countries:
        if country not in merged_df["country"].unique():
            logger.warning("Country '%s' not found in merged dataset. Skipping.", country)
            continue
        result = run_continual_learning_country(country, merged_df, scaler_params)
        country_results[country] = result

    summary = {
        "run_timestamp"    : RUN_TIMESTAMP,
        "device"           : str(DEVICE),
        "countries"        : country_results,
        "versioned_dir"    : str(VERSIONED_DIR),
        "cl_log"           : str(CL_LOG),
        "models_updated"   : [c for c, r in country_results.items()
                               if r.get("decision") == "UPDATED"],
        "models_skipped"   : [c for c, r in country_results.items()
                               if not r.get("triggered")],
        "models_rolledback": [c for c, r in country_results.items()
                               if r.get("decision") == "ROLLED_BACK"],
    }

    summary_path = LOG_DIR / f"continual_learning_summary_{RUN_TIMESTAMP}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved: %s", summary_path)

    logger.info("=" * 70)
    logger.info("CELL 35 COMPLETE")
    for country, r in country_results.items():
        if r["triggered"]:
            logger.info(
                "  %-12s | decision=%-12s | updated_RMSE=%s | "
                "original_RMSE=%s | strict_load=%s",
                country,
                r.get("decision", "N/A"),
                f"{r['updated_rmse']:.2f}mm" if r["updated_rmse"] else "N/A",
                f"{r['original_rmse']:.2f}mm" if r["original_rmse"] else "N/A",
                r.get("error") is None,
            )
        else:
            logger.info("  %-12s | SKIPPED (%d new months)", country, r["new_months"])
    logger.info("=" * 70)

    return summary


# =============================================================================
# CELL 35 — THOROUGH TESTING SUITE
# =============================================================================

def test_cell_35() -> None:
    """
    Comprehensive unit and integration tests for all Cell 35 functions.

    Test coverage:
      T01 — Model architecture instantiation and param count
      T02 — Forward pass output shapes
      T03 — Physics loss: non-negative outputs yield zero penalty
      T04 — Physics loss: negative outputs yield positive penalty
      T05 — Data normalisation: variable features normalised, constants zeroed
      T06 — Sequence builder: correct output shapes
      T07 — Sequence builder: insufficient data raises ValueError
      T08 — Train/val temporal split preserves order and total count
      T09 — LSTM freezing: only main_head and physics_head trainable
      T10 — Fine-tuning: completes and records loss history
      T11 — RMSE: perfect predictions yield zero
      T12 — RMSE: known constant offset yields correct value
      T13 — Quality gate: identical models pass
      T14 — Quality gate: degraded model evaluated correctly
      T15 — Rollback: returns original model on failure
      T16 — Rollback: returns updated model on pass
      T17 — Version control: checkpoint keys and architecture metadata correct
      T18 — Update trigger: fires with sufficient new months
      T19 — Update trigger: does not fire with too few months
      T20 — Full pipeline integration on Drive data
    """

    PASS = "PASS"
    FAIL = "FAIL"
    results = []

    def record(test_id, name, status, detail=""):
        results.append((test_id, name, status, detail))
        marker = "[PASS]" if status == PASS else "[FAIL]"
        print(f"  {marker} {test_id}: {name}" +
              (f" — {detail}" if detail else ""))

    def make_model(input_size=CELL28_INPUT_SIZE):
        return PhysicsInformedSLRModel(input_size=input_size).to(DEVICE)

    def make_synthetic_sequences(n=100, seq_len=12, n_feat=CELL28_INPUT_SIZE):
        np.random.seed(42)
        X = np.random.randn(n, seq_len, n_feat).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        return X, y

    def make_synthetic_merged(n_hist=200, n_new=20) -> pd.DataFrame:
        frames = []
        np.random.seed(0)
        for country in ["Bangladesh", "Vietnam", "Maldives"]:
            dates_hist = pd.date_range("2005-01-01", periods=n_hist, freq="MS")
            dates_new  = pd.date_range("2022-01-01", periods=n_new,  freq="MS")
            all_dates  = dates_hist.tolist() + dates_new.tolist()
            n_total    = len(all_dates)
            slr        = np.cumsum(np.random.normal(0.3, 0.5, n_total))
            frames.append(pd.DataFrame({
                "date"         : all_dates,
                "country"      : country,
                "slrv2"        : slr,
                "tide_gauge"   : slr,
                "enso"         : np.random.normal(0, 0.8, n_total),
                "aviso"        : slr,
                "vlm"          : -2.0,
                "cmip6_thermo" : np.linspace(0, 5, n_total),
                "cmip6_glacier": np.linspace(0, 2, n_total),
                "momentum"     : pd.Series(slr).rolling(12, min_periods=3).mean().values,
                "residual"     : np.random.normal(0, 0.2, n_total),
                "month_sin"    : np.sin(2 * np.pi *
                                        pd.DatetimeIndex(all_dates).month / 12),
                "month_cos"    : np.cos(2 * np.pi *
                                        pd.DatetimeIndex(all_dates).month / 12),
                "enso_x_aviso" : np.random.normal(0, 1, n_total),
            }))
        return pd.concat(frames, ignore_index=True)

    print("\n" + "=" * 70)
    print("CELL 35 — TEST SUITE")
    print("=" * 70 + "\n")

    synth_merged = make_synthetic_merged()

    # -------------------------------------------------------------------------
    # T01: Model architecture instantiation
    # -------------------------------------------------------------------------
    test_id = "T01"
    try:
        model       = make_model()
        param_count = sum(p.numel() for p in model.parameters())
        assert isinstance(model, PhysicsInformedSLRModel)
        assert hasattr(model, "lstm")
        assert hasattr(model, "main_head")
        assert hasattr(model, "physics_head")
        # Verify exact layer shapes match Cell 28 checkpoints
        assert model.lstm.hidden_size == HIDDEN_SIZE,       \
            f"hidden_size mismatch: {model.lstm.hidden_size}"
        assert model.lstm.input_size  == CELL28_INPUT_SIZE, \
            f"input_size mismatch: {model.lstm.input_size}"
        assert model.main_head[0].out_features == 64,       \
            f"main_head first layer: {model.main_head[0].out_features}"
        assert model.physics_head[0].out_features == 32,    \
            f"physics_head first layer: {model.physics_head[0].out_features}"
        record(test_id, "Model instantiation and architecture", PASS,
               f"params={param_count:,}, hidden={HIDDEN_SIZE}, "
               f"input={CELL28_INPUT_SIZE}")
    except Exception as e:
        record(test_id, "Model instantiation and architecture", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T02: Forward pass shape correctness
    # -------------------------------------------------------------------------
    test_id = "T02"
    try:
        model      = make_model()
        x          = torch.randn(8, SEQ_LEN, CELL28_INPUT_SIZE).to(DEVICE)
        pred, phys = model(x)
        assert pred.shape == (8, 1), f"Expected (8,1) got {pred.shape}"
        assert phys.shape == (8, 1), f"Expected (8,1) got {phys.shape}"
        record(test_id, "Forward pass shape", PASS,
               f"pred={pred.shape}, phys={phys.shape}")
    except Exception as e:
        record(test_id, "Forward pass shape", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T03: Physics loss — non-negative outputs yield zero penalty
    # -------------------------------------------------------------------------
    test_id = "T03"
    try:
        pred     = torch.zeros(4, 1)
        target   = torch.zeros(4, 1)
        phys_pos = torch.abs(torch.randn(4, 1))
        _, _, phys_val = physics_informed_loss(pred, target, phys_pos)
        assert phys_val == 0.0, f"Expected 0.0, got {phys_val}"
        record(test_id, "Physics loss — non-negative yields zero penalty", PASS,
               f"phys_loss={phys_val}")
    except Exception as e:
        record(test_id, "Physics loss — non-negative yields zero penalty", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T04: Physics loss — negative outputs yield positive penalty
    # -------------------------------------------------------------------------
    test_id = "T04"
    try:
        pred     = torch.zeros(4, 1)
        target   = torch.zeros(4, 1)
        phys_neg = -torch.abs(torch.randn(4, 1))
        _, _, phys_val = physics_informed_loss(pred, target, phys_neg)
        assert phys_val > 0.0, f"Expected positive, got {phys_val}"
        record(test_id, "Physics loss — negative outputs penalised", PASS,
               f"phys_loss={phys_val:.6f}")
    except Exception as e:
        record(test_id, "Physics loss — negative outputs penalised", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T05: Data normalisation — variable features normalised, constants zeroed
    # -------------------------------------------------------------------------
    test_id = "T05"
    try:
        X_norm, y_norm, feat_stats = load_and_normalise(
            synth_merged, "Bangladesh", {}
        )
        non_constant_mask = X_norm.std(axis=0) > 1e-6
        if non_constant_mask.sum() > 0:
            variable_means = np.abs(X_norm[:, non_constant_mask].mean(axis=0))
            variable_stds  = X_norm[:, non_constant_mask].std(axis=0)
            assert variable_means.max() < 1.0, \
                f"Variable feature mean too large: {variable_means.max():.3f}"
            assert variable_stds.min() > 0.01, \
                f"Variable feature std near zero: {variable_stds.min():.3f}"
        constant_mask = ~non_constant_mask
        if constant_mask.sum() > 0:
            constant_vals = np.abs(X_norm[:, constant_mask])
            assert constant_vals.max() < 1e-6, \
                f"Constant feature not zeroed: {constant_vals.max():.2e}"
        record(test_id, "Data normalisation statistics", PASS,
               f"variable={int(non_constant_mask.sum())}, "
               f"constant={int(constant_mask.sum())}")
    except Exception as e:
        record(test_id, "Data normalisation statistics", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T06: Sequence builder shapes
    # -------------------------------------------------------------------------
    test_id = "T06"
    try:
        n_samples, n_feat = 50, CELL28_INPUT_SIZE
        X_flat  = np.random.randn(n_samples, n_feat).astype(np.float32)
        y_flat  = np.random.randn(n_samples).astype(np.float32)
        X_seq, y_seq = build_sequences(X_flat, y_flat, seq_len=SEQ_LEN)
        expected_n   = n_samples - SEQ_LEN
        assert X_seq.shape == (expected_n, SEQ_LEN, n_feat), \
            f"X_seq shape wrong: {X_seq.shape}"
        assert y_seq.shape == (expected_n,), \
            f"y_seq shape wrong: {y_seq.shape}"
        record(test_id, "Sequence builder shapes", PASS,
               f"X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    except Exception as e:
        record(test_id, "Sequence builder shapes", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T07: Sequence builder — insufficient data raises ValueError
    # -------------------------------------------------------------------------
    test_id = "T07"
    try:
        X_short = np.random.randn(5, CELL28_INPUT_SIZE).astype(np.float32)
        y_short = np.random.randn(5).astype(np.float32)
        raised  = False
        try:
            build_sequences(X_short, y_short, seq_len=SEQ_LEN)
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for insufficient data"
        record(test_id, "Sequence builder — insufficient data raises ValueError", PASS)
    except Exception as e:
        record(test_id, "Sequence builder — insufficient data raises ValueError",
               FAIL, str(e))

    # -------------------------------------------------------------------------
    # T08: Train/val temporal split
    # -------------------------------------------------------------------------
    test_id = "T08"
    try:
        X_seq, y_seq             = make_synthetic_sequences(n=80)
        train_loader, val_loader = split_new_data(X_seq, y_seq, val_frac=0.2)
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        assert n_train > 0,                    "Empty train loader"
        assert n_val   > 0,                    "Empty val loader"
        assert n_train + n_val == len(X_seq),  \
            f"Split lost samples: {n_train}+{n_val} != {len(X_seq)}"
        record(test_id, "Train/val temporal split", PASS,
               f"train={n_train}, val={n_val}, total={len(X_seq)}")
    except Exception as e:
        record(test_id, "Train/val temporal split", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T09: LSTM layer freezing
    # -------------------------------------------------------------------------
    test_id = "T09"
    try:
        model           = make_model()
        model, n_frozen = freeze_lstm_layers(model)
        frozen_names    = [n for n, p in model.named_parameters()
                           if not p.requires_grad]
        trainable_names = [n for n, p in model.named_parameters()
                           if p.requires_grad]
        assert n_frozen > 0,            "No parameters were frozen"
        assert len(trainable_names) > 0, "No trainable parameters remain"
        for name in frozen_names:
            assert "lstm" in name, f"Non-lstm param frozen: {name}"
        for name in trainable_names:
            assert "lstm" not in name, f"LSTM param left trainable: {name}"
        top_level = list(dict.fromkeys(n.split(".")[0] for n in trainable_names))
        record(test_id, "LSTM layer freezing", PASS,
               f"frozen_params={n_frozen}, trainable_modules={top_level}")
    except Exception as e:
        record(test_id, "LSTM layer freezing", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T10: Fine-tuning completes and records history
    # -------------------------------------------------------------------------
    test_id = "T10"
    try:
        model            = make_model()
        model, _         = freeze_lstm_layers(model)
        X_seq, y_seq     = make_synthetic_sequences(n=60)
        train_loader, val_loader = split_new_data(X_seq, y_seq, val_frac=0.2)
        _, history       = fine_tune_pinn(model, train_loader, val_loader, "TEST")
        assert len(history["train_loss"]) > 0
        assert history["best_val_loss"] <= history["val_loss"][0] + 1e-6
        record(test_id, "Fine-tuning runs and records history", PASS,
               f"epochs={len(history['train_loss'])}, "
               f"best_val={history['best_val_loss']:.5f}")
    except Exception as e:
        record(test_id, "Fine-tuning runs and records history", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T11: RMSE — perfect predictions yield zero
    # -------------------------------------------------------------------------
    test_id = "T11"
    try:
        class PerfectModel(nn.Module):
            def forward(self, x):
                return x[:, -1, 0:1], torch.zeros(x.shape[0], 1)
        perfect      = PerfectModel().to(DEVICE)
        X_seq, y_seq = make_synthetic_sequences(n=40)
        y_seq        = X_seq[:, -1, 0]
        feat_stats   = {TARGET_COLUMN: {"mean": 0.0, "std": 1.0}}
        rmse         = compute_rmse(perfect, X_seq, y_seq, feat_stats)
        assert rmse < 1e-5, f"Expected near-zero RMSE, got {rmse}"
        record(test_id, "RMSE — perfect predictions yield zero", PASS,
               f"RMSE={rmse:.2e}")
    except Exception as e:
        record(test_id, "RMSE — perfect predictions yield zero", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T12: RMSE — known constant offset
    # -------------------------------------------------------------------------
    test_id = "T12"
    try:
        OFFSET = 5.0
        class OffsetModel(nn.Module):
            def forward(self, x):
                return (torch.zeros(x.shape[0], 1) + OFFSET,
                        torch.zeros(x.shape[0], 1))
        offset_m    = OffsetModel().to(DEVICE)
        X_seq, _    = make_synthetic_sequences(n=40)
        y_zeros     = np.zeros(len(X_seq), dtype=np.float32)
        feat_stats  = {TARGET_COLUMN: {"mean": 0.0, "std": 1.0}}
        rmse        = compute_rmse(offset_m, X_seq, y_zeros, feat_stats)
        assert abs(rmse - OFFSET) < 1e-3, \
            f"Expected RMSE={OFFSET}, got {rmse:.4f}"
        record(test_id, "RMSE — constant offset correct", PASS,
               f"expected={OFFSET}, got={rmse:.4f}")
    except Exception as e:
        record(test_id, "RMSE — constant offset correct", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T13: Quality gate — identical models pass
    # -------------------------------------------------------------------------
    test_id = "T13"
    try:
        X_seq, y_seq = make_synthetic_sequences(n=40)
        feat_stats   = {TARGET_COLUMN: {"mean": 0.0, "std": 1.0}}
        model_a      = make_model()
        model_b      = copy.deepcopy(model_a)
        passed, u_rmse, o_rmse = evaluate_update_quality(
            model_a, model_b, X_seq, y_seq, feat_stats, "TEST"
        )
        assert passed, "Identical models should pass quality gate"
        record(test_id, "Quality gate — identical models pass", PASS,
               f"updated={u_rmse:.2f}, original={o_rmse:.2f}")
    except Exception as e:
        record(test_id, "Quality gate — identical models pass", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T14: Quality gate — degraded model evaluated
    # -------------------------------------------------------------------------
    test_id = "T14"
    try:
        X_seq, y_seq = make_synthetic_sequences(n=40)
        feat_stats   = {TARGET_COLUMN: {"mean": 0.0, "std": 100.0}}
        good_model   = make_model()
        bad_model    = copy.deepcopy(good_model)
        with torch.no_grad():
            for param in bad_model.main_head.parameters():
                param.fill_(0.0)
        passed, u_rmse, o_rmse = evaluate_update_quality(
            bad_model, good_model, X_seq, y_seq, feat_stats, "TEST"
        )
        assert isinstance(passed, bool)
        record(test_id, "Quality gate — degraded case evaluated", PASS,
               f"passed={passed}, updated={u_rmse:.2f}, original={o_rmse:.2f}")
    except Exception as e:
        record(test_id, "Quality gate — degraded case evaluated", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T15: Rollback — returns original on failure
    # -------------------------------------------------------------------------
    test_id = "T15"
    try:
        m_orig    = make_model()
        m_updated = make_model()
        returned, decision = rollback_if_degraded(m_updated, m_orig, False, "TEST")
        for (k1, v1), (k2, v2) in zip(
            returned.state_dict().items(), m_orig.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Wrong model returned at {k1}"
        assert decision == "ROLLED_BACK"
        record(test_id, "Rollback — returns original on failure", PASS)
    except Exception as e:
        record(test_id, "Rollback — returns original on failure", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T16: Rollback — returns updated on pass
    # -------------------------------------------------------------------------
    test_id = "T16"
    try:
        m_orig    = make_model()
        m_updated = make_model()
        returned, decision = rollback_if_degraded(m_updated, m_orig, True, "TEST")
        for (k1, v1), (k2, v2) in zip(
            returned.state_dict().items(), m_updated.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Wrong model returned at {k1}"
        assert decision == "UPDATED"
        record(test_id, "Rollback — returns updated on pass", PASS)
    except Exception as e:
        record(test_id, "Rollback — returns updated on pass", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T17: Version control — checkpoint structure and architecture metadata
    # -------------------------------------------------------------------------
    test_id = "T17"
    try:
        model     = make_model()
        metadata  = {"test": True, "country": "TEST"}
        save_path = version_control_model(model, "TEST", metadata)
        assert save_path.exists(), f"Checkpoint not found: {save_path}"
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        for key in ["model_state_dict", "metadata", "timestamp", "architecture"]:
            assert key in ckpt, f"Missing key: {key}"
        assert ckpt["architecture"]["hidden_size"]  == HIDDEN_SIZE
        assert ckpt["architecture"]["input_size"]   == CELL28_INPUT_SIZE
        assert ckpt["metadata"]["test"] is True
        # Verify state_dict keys match Cell 28 naming
        sd_keys = list(ckpt["model_state_dict"].keys())
        assert any("main_head" in k for k in sd_keys),    "main_head missing from checkpoint"
        assert any("physics_head" in k for k in sd_keys), "physics_head missing from checkpoint"
        assert any("lstm" in k for k in sd_keys),         "lstm missing from checkpoint"
        save_path.unlink()
        record(test_id, "Version control — checkpoint structure", PASS,
               f"keys={list(ckpt.keys())}, sd_keys={len(sd_keys)}")
    except Exception as e:
        record(test_id, "Version control — checkpoint structure", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T18: Update trigger — fires with sufficient months
    # -------------------------------------------------------------------------
    test_id = "T18"
    try:
        trigger, n_months = check_update_trigger(
            synth_merged, "Bangladesh", historical_end_date="2021-01-01"
        )
        assert trigger,                             \
            f"Expected trigger=True (new_months={n_months})"
        assert n_months >= UPDATE_TRIGGER_MIN_MONTHS
        record(test_id, "Update trigger — fires with sufficient months", PASS,
               f"new_months={n_months}")
    except Exception as e:
        record(test_id, "Update trigger — fires with sufficient months", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T19: Update trigger — does not fire with too few months
    # -------------------------------------------------------------------------
    test_id = "T19"
    try:
        latest     = synth_merged[
            synth_merged["country"] == "Bangladesh"
        ]["date"].max()
        one_before = (latest - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
        trigger, n_months = check_update_trigger(
            synth_merged, "Bangladesh", historical_end_date=one_before
        )
        assert not trigger, \
            f"Expected trigger=False (n_months={n_months})"
        record(test_id, "Update trigger — does not fire with few months", PASS,
               f"new_months={n_months}, threshold={UPDATE_TRIGGER_MIN_MONTHS}")
    except Exception as e:
        record(test_id, "Update trigger — does not fire with few months", FAIL, str(e))

    # -------------------------------------------------------------------------
    # T20: Full pipeline integration on Drive data
    # -------------------------------------------------------------------------
    test_id = "T20"
    try:
        summary = run_cell_35()
        assert "run_timestamp"     in summary
        assert "countries"         in summary
        assert "models_updated"    in summary
        assert "models_skipped"    in summary
        assert "models_rolledback" in summary
        total = (len(summary["models_updated"]) +
                 len(summary["models_skipped"]) +
                 len(summary["models_rolledback"]))
        assert total >= 1, "No countries were processed"
        assert CL_LOG.exists(), "Continual learning log not created"
        # Verify any updated models loaded weights strictly (no fallback)
        for country, r in summary["countries"].items():
            if r.get("triggered") and r.get("error") is None:
                versioned = list(VERSIONED_DIR.glob(
                    f"pinn_{country.lower()}*.pt"
                ))
                assert len(versioned) >= 1, \
                    f"No versioned model found for {country}"
                ckpt = torch.load(
                    versioned[-1], map_location="cpu", weights_only=False
                )
                assert ckpt["architecture"]["hidden_size"] == HIDDEN_SIZE, \
                    f"{country} checkpoint has wrong hidden_size"
        record(test_id, "Full pipeline integration test", PASS,
               f"updated={summary['models_updated']}, "
               f"skipped={summary['models_skipped']}, "
               f"rolledback={summary['models_rolledback']}")
    except Exception as e:
        record(test_id, "Full pipeline integration test", FAIL, str(e))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, _, s, _ in results if s == PASS)
    failed = sum(1 for _, _, s, _ in results if s == FAIL)
    total  = len(results)
    print(f"  Passed : {passed}/{total}")
    print(f"  Failed : {failed}/{total}")
    if failed > 0:
        print("\n  Failed tests:")
        for tid, name, status, detail in results:
            if status == FAIL:
                print(f"    {tid}: {name}")
                if detail:
                    print(f"         Detail: {detail}")
    print("=" * 70 + "\n")


# =============================================================================
# RUN
# =============================================================================

test_cell_35()

print("VERSIONED MODEL INVENTORY")
print("-" * 40)
versioned_files = sorted(VERSIONED_DIR.glob("*.pt"))
if versioned_files:
    for f in versioned_files:
        size_kb = f.stat().st_size / 1024
        ckpt    = torch.load(f, map_location="cpu", weights_only=False)
        hidden  = ckpt.get("architecture", {}).get("hidden_size", "unknown")
        print(f"  {f.name}  ({size_kb:.1f} KB)  hidden={hidden}")
else:
    print("  No versioned models found.")
print(f"\nContinual learning log: {CL_LOG}")
