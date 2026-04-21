from pathlib import Path
from datetime import datetime
import yaml

from src.c2dti.config_validation import validate_config
from src.c2dti.output_io import make_run_dir, write_summary, write_config_snapshot, append_registry, write_prediction_matrix
from src.c2dti.causal_objective import compute_causal_score, compute_causal_reliability_score
from src.c2dti.data_utils import summarize_matrix
from src.c2dti.dataset_loader import load_dti_dataset
from src.c2dti.dti_model import create_predictor
from src.c2dti.evaluation import evaluate_predictions
from src.c2dti.perturbation import perturb_dataset_interactions
from src.c2dti.splitter import split_dataset
def dry_run(config_path: str) -> int:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    errors = validate_config(cfg)
    if errors:
        print("[ERROR] Config validation failed:")
        for e in errors:
            print(f"- {e}")
        return 2

    print("[OK] Dry-run passed")
    print(f"name={cfg.get('name')}")
    print(f"protocol={cfg.get('protocol')}")
    print(f"output.base_dir={cfg.get('output', {}).get('base_dir')}")
    if cfg.get("dataset"):
        print(f"dataset.name={cfg.get('dataset', {}).get('name')}")
        print(f"dataset.path={cfg.get('dataset', {}).get('path')}")
    if cfg.get("model"):
        print(f"model.name={cfg.get('model', {}).get('name', 'simple_baseline')}")
    print(f"config={cfg_path}")
    return 0

def run_once(config_path: str) -> int:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 1

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    errors = validate_config(cfg)
    if errors:
        print("[ERROR] Config validation failed:")
        for e in errors:
            print(f"- {e}")
        return 2

    name = cfg.get("name", "unnamed")
    protocol = cfg.get("protocol", "P0")
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")

    run_dir = make_run_dir(base_dir, name)
    config_snapshot = write_config_snapshot(run_dir, cfg)

    causal_cfg = cfg.get("causal", {})
    causal_enabled = causal_cfg.get("enabled", False)
    causal_weight = causal_cfg.get("weight", 0.0)

    summary_payload = {
        "run_name": name,
        "protocol": protocol,
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Minimal run contract smoke step (no model training yet)."
    }

    dataset_cfg = cfg.get("dataset")
    if dataset_cfg:
        dataset = load_dti_dataset(dataset_cfg["name"], Path(dataset_cfg["path"]))
        allow_placeholder = dataset_cfg.get("allow_placeholder", True)
        if bool(dataset.metadata.get("is_placeholder", False)) and not allow_placeholder:
            print("[ERROR] Dataset placeholder was used but dataset.allow_placeholder is false")
            print(f"[ERROR] Please provide real files at: {dataset_cfg.get('path')}")
            return 3

        # Pass the full model config dict so MatrixFactorizationDTIPredictor
        # can read latent_dim, epochs, lr, seed from the YAML file.
        predictor = create_predictor(cfg.get("model", {}))
        # --- Train/test split ---
        # If a split config is present, divide the known pairs into train and test sets.
        # The model trains only on train-set entries; metrics are reported on test-set
        # entries only (the held-out pairs the model never saw during training).
        # If no split config is present, the old behaviour is preserved: train and
        # evaluate on all known pairs (useful for smoke tests and dry-runs).
        split_cfg = cfg.get("split")
        train_mask = None
        test_mask = None
        if split_cfg and not dataset.metadata.get("is_placeholder", False):
            train_mask, test_mask = split_dataset(
                dataset,
                strategy=split_cfg.get("strategy", "random"),
                test_ratio=float(split_cfg.get("test_ratio", 0.2)),
                seed=int(split_cfg.get("seed", 42)),
            )
            summary_payload["split"] = {
                "strategy": split_cfg.get("strategy", "random"),
                "test_ratio": float(split_cfg.get("test_ratio", 0.2)),
                "seed": int(split_cfg.get("seed", 42)),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }

        # Train the model (passing train_mask so it never peeks at test entries)
        predictions = predictor.fit_predict(dataset, train_mask=train_mask)
        prediction_path = write_prediction_matrix(run_dir, dataset.drugs, dataset.targets, predictions)

        summary_payload["notes"] = "Real DTI pipeline completed with dataset loading, split, prediction, evaluation, and optional causal reliability."
        summary_payload["dataset_name"] = dataset.metadata.get("source", dataset_cfg["name"])
        summary_payload["dataset_placeholder"] = bool(dataset.metadata.get("is_placeholder", False))
        summary_payload["dataset_allow_placeholder"] = allow_placeholder
        summary_payload["num_drugs"] = len(dataset.drugs)
        summary_payload["num_targets"] = len(dataset.targets)
        summary_payload["prediction_path"] = str(prediction_path)
        summary_payload["prediction_stats"] = summarize_matrix(predictions)

        # --- Evaluation ---
        # When a split was performed: evaluate ONLY on held-out test-set pairs.
        # This gives scientifically valid metrics comparable to published results.
        # When no split: evaluate on all known pairs (backward-compatible).
        if test_mask is not None:
            # Extract true and predicted values for test-set pairs only
            y_true_test = dataset.interactions[test_mask]
            y_pred_test = predictions[test_mask]
            summary_payload["evaluation_metrics"] = evaluate_predictions(y_true_test, y_pred_test)

            # Also record training-set fit quality so we can detect overfitting
            y_true_train = dataset.interactions[train_mask]
            y_pred_train = predictions[train_mask]
            summary_payload["train_metrics"] = evaluate_predictions(y_true_train, y_pred_train)
        else:
            # No split: evaluate on all known pairs (smoke-test / placeholder mode)
            summary_payload["evaluation_metrics"] = evaluate_predictions(
                dataset.interactions, predictions
            )

        # If the predictor was trainable, record its loss curve and save the checkpoint.
        if hasattr(predictor, "train_loss_history") and predictor.train_loss_history:
            # Store first, last, and min loss so the summary is human-readable
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

        if causal_enabled:
            perturbation_cfg = cfg.get("perturbation", {})
            perturbed_dataset = perturb_dataset_interactions(
                dataset,
                strength=perturbation_cfg.get("strength", 0.1),
                seed=perturbation_cfg.get("seed", 42),
            )
            perturbed_predictions = predictor.fit_predict(perturbed_dataset)
            summary_payload["causal_score"] = compute_causal_reliability_score(
                baseline_predictions=predictions,
                perturbed_predictions=perturbed_predictions,
                weight=causal_weight if causal_weight > 0 else 1.0,
            )
    else:
        causal_score = compute_causal_score(enabled=causal_enabled, weight=causal_weight)
        if causal_score is not None:
            summary_payload["causal_score"] = causal_score
    
    summary_path = write_summary(run_dir, summary_payload)

    append_registry(
        base_dir=base_dir,
        row={
            "run_name": name,
            "protocol": protocol,
            "status": "completed",
            "summary_path": str(summary_path),
            "config_snapshot_path": str(config_snapshot),
            "created_at": summary_payload["created_at"],
        },
    )

    print("[OK] Run contract completed")
    print(f"run_dir={run_dir}")
    print(f"summary={summary_path}")
    print(f"config_snapshot={config_snapshot}")
    print(f"registry={Path(base_dir) / 'results_registry.csv'}")
    return 0
