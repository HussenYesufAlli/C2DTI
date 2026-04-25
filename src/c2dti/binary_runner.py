"""Binary classification runner for C2DTI.

This runner is intentionally separate from the regression runner so existing
continuous prediction behavior remains unchanged.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.c2dti.binary_dataset_loader import load_binary_dti_dataset
from src.c2dti.binary_evaluation import evaluate_binary_predictions
from src.c2dti.causal_runtime import compute_causal_outputs
from src.c2dti.config_validation import validate_config
from src.c2dti.data_utils import summarize_matrix
from src.c2dti.dti_model import create_predictor
from src.c2dti.output_io import (
    append_registry,
    make_run_dir,
    write_config_snapshot,
    write_prediction_matrix,
    write_summary,
)
from src.c2dti.splitter import split_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_runtime_path(raw_path: str) -> Path:
    # Resolve relative config paths against the C2DTI repository root.
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _validate_binary_section(cfg: dict) -> list[str]:
    # Validate binary-specific options without changing global validation logic.
    binary_cfg = cfg.get("binary", {})
    if binary_cfg is None:
        binary_cfg = {}

    if not isinstance(binary_cfg, dict):
        return ["binary config must be a mapping"]

    errors: list[str] = []
    threshold = binary_cfg.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
        errors.append("binary.threshold must be a number in [0, 1]")

    return errors


def dry_run_binary(config_path: str) -> int:
    # Perform validation-only checks for binary mode before executing a run.
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    errors = validate_config(cfg)
    errors.extend(_validate_binary_section(cfg))
    if errors:
        print("[ERROR] Binary config validation failed:")
        for err in errors:
            print(f"- {err}")
        return 2

    dataset_cfg = cfg.get("dataset", {})
    print("[OK] Binary dry-run passed")
    print(f"name={cfg.get('name')}")
    print(f"protocol={cfg.get('protocol')}")
    print(f"dataset.name={dataset_cfg.get('name')}")
    print(f"dataset.path={_resolve_runtime_path(str(dataset_cfg.get('path', '')))}")
    print(f"binary.threshold={cfg.get('binary', {}).get('threshold', 0.5)}")
    return 0


def run_once_binary(config_path: str) -> int:
    """Run one full binary DTI experiment.

    This keeps the same run artifact structure as regression mode, but writes
    binary metrics in summary.json.
    """

    # Load and validate config first to fail early and clearly.
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    errors = validate_config(cfg)
    errors.extend(_validate_binary_section(cfg))
    if errors:
        print("[ERROR] Binary config validation failed:")
        for err in errors:
            print(f"- {err}")
        return 2

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "")).strip()
    dataset_path = _resolve_runtime_path(str(dataset_cfg.get("path", "")))
    allow_placeholder = bool(dataset_cfg.get("allow_placeholder", True))

    dataset = load_binary_dti_dataset(dataset_name=dataset_name, csv_path=dataset_path)
    if bool(dataset.metadata.get("is_placeholder", False)) and not allow_placeholder:
        print("[ERROR] Binary fallback dataset was used but strict dataset mode is enabled")
        return 3

    # Build run artifact directory and snapshot before training for reproducibility.
    run_name = str(cfg.get("name", "unnamed_binary_run"))
    base_dir = str(_resolve_runtime_path(str(cfg.get("output", {}).get("base_dir", "outputs"))))
    run_dir = make_run_dir(base_dir, run_name)
    config_snapshot = write_config_snapshot(run_dir, cfg)

    split_cfg = cfg.get("split") or {}
    train_mask = None
    test_mask = None
    if split_cfg:
        train_mask, test_mask = split_dataset(
            dataset=dataset,
            strategy=str(split_cfg.get("strategy", "random")),
            test_ratio=float(split_cfg.get("test_ratio", 0.2)),
            seed=int(split_cfg.get("seed", 42)),
        )

    # Reuse existing predictor implementations to output probabilities/scores.
    predictor = create_predictor(cfg.get("model", {}))
    predictions = predictor.fit_predict(dataset, train_mask=train_mask)
    prediction_path = write_prediction_matrix(run_dir, dataset.drugs, dataset.targets, predictions)

    threshold = float((cfg.get("binary") or {}).get("threshold", 0.5))

    # Evaluate on test set when split exists; otherwise evaluate all known labels.
    if test_mask is not None:
        y_true_eval = dataset.interactions[test_mask]
        y_score_eval = predictions[test_mask]
        eval_metrics = evaluate_binary_predictions(y_true_eval, y_score_eval, threshold=threshold)

        y_true_train = dataset.interactions[train_mask]
        y_score_train = predictions[train_mask]
        train_metrics = evaluate_binary_predictions(y_true_train, y_score_train, threshold=threshold)
    else:
        known_mask = ~np.isnan(dataset.interactions)
        y_true_eval = dataset.interactions[known_mask]
        y_score_eval = predictions[known_mask]
        eval_metrics = evaluate_binary_predictions(y_true_eval, y_score_eval, threshold=threshold)
        train_metrics = None

    # Persist a summary compatible with existing output/report scripts.
    summary_payload = {
        "run_name": run_name,
        "protocol": str(cfg.get("protocol", "P_binary")),
        "task_type": "binary_classification",
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset.metadata.get("source", dataset_name),
        "dataset_placeholder": bool(dataset.metadata.get("is_placeholder", False)),
        "dataset_allow_placeholder": allow_placeholder,
        "num_drugs": len(dataset.drugs),
        "num_targets": len(dataset.targets),
        "prediction_path": str(prediction_path),
        "prediction_stats": summarize_matrix(predictions),
        "binary_threshold": threshold,
        "evaluation_metrics": eval_metrics,
        "config_snapshot": str(config_snapshot),
    }

    if split_cfg:
        summary_payload["split"] = {
            "strategy": str(split_cfg.get("strategy", "random")),
            "test_ratio": float(split_cfg.get("test_ratio", 0.2)),
            "seed": int(split_cfg.get("seed", 42)),
            "n_train": int(train_mask.sum()) if train_mask is not None else 0,
            "n_test": int(test_mask.sum()) if test_mask is not None else 0,
        }

    if train_metrics is not None:
        summary_payload["train_metrics"] = train_metrics

    if hasattr(predictor, "train_loss_history") and getattr(predictor, "train_loss_history", None):
        history = predictor.train_loss_history
        summary_payload["training"] = {
            "epochs_completed": len(history),
            "loss_start": round(history[0], 6),
            "loss_final": round(history[-1], 6),
            "loss_min": round(min(history), 6),
        }

    if hasattr(predictor, "save_checkpoint"):
        checkpoint_path = predictor.save_checkpoint(run_dir)
        summary_payload["checkpoint_path"] = str(checkpoint_path)

    causal_cfg = cfg.get("causal", {}) or {}
    causal_enabled = bool(causal_cfg.get("enabled", False))
    if causal_enabled:
        causal_output = compute_causal_outputs(
            dataset=dataset,
            predictions=predictions,
            train_mask=train_mask,
            causal_cfg=causal_cfg,
            perturbation_cfg=cfg.get("perturbation", {}),
            predictor=predictor,
        )
        summary_payload["causal"] = causal_output["causal"]
        summary_payload["causal_score"] = causal_output["causal_score"]

    summary_path = write_summary(run_dir, summary_payload)
    append_registry(base_dir, {
        "run_name": run_name,
        "protocol": str(cfg.get("protocol", "P_binary")),
        "status": "completed",
        "summary_path": str(summary_path),
        "config_snapshot_path": str(config_snapshot),
        "created_at": summary_payload["created_at"],
    })

    # Print compact terminal report for quick experiment tracking.
    print(f"[OK] Binary run completed: {run_name}")
    print(f"[OK] Summary: {summary_path}")
    print(
        "[OK] Metrics "
        f"AUROC={eval_metrics.get('auroc')} "
        f"AUPRC={eval_metrics.get('auprc')} "
        f"F1={eval_metrics.get('f1')} "
        f"ACC={eval_metrics.get('accuracy')} "
        f"SEN={eval_metrics.get('sensitivity')} "
        f"SPE={eval_metrics.get('specificity')}"
    )

    return 0
