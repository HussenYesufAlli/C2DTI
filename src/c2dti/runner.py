from pathlib import Path
from datetime import datetime
import yaml

from src.c2dti.config_validation import validate_config
from src.c2dti.output_io import make_run_dir, write_summary, write_config_snapshot, append_registry, write_prediction_matrix
from src.c2dti.causal_objective import (
    compute_causal_score,
    compute_causal_reliability_score,
    compute_cross_view_causal_metrics,
    compute_mas_losses,
    compute_irm_cf_losses,
)
from src.c2dti.data_utils import summarize_matrix
from src.c2dti.dataset_loader import load_dti_dataset
from src.c2dti.dti_model import create_predictor
from src.c2dti.evaluation import evaluate_predictions
from src.c2dti.perturbation import perturb_dataset_interactions
from src.c2dti.splitter import split_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_runtime_path(raw_path: str) -> Path:
    """Resolve config paths relative to the C2DTI repository root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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
    output_dir = cfg.get('output', {}).get('base_dir')
    if output_dir:
        print(f"output.base_dir={_resolve_runtime_path(str(output_dir))}")
    if cfg.get("dataset"):
        dataset_path = cfg.get('dataset', {}).get('path')
        print(f"dataset.name={cfg.get('dataset', {}).get('name')}")
        if dataset_path:
            print(f"dataset.path={_resolve_runtime_path(str(dataset_path))}")
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
    base_dir = str(_resolve_runtime_path(str(cfg.get("output", {}).get("base_dir", "outputs"))))

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
        dataset = load_dti_dataset(dataset_cfg["name"], _resolve_runtime_path(str(dataset_cfg["path"])))
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
            causal_mode = str(causal_cfg.get("mode", "reliability")).strip().lower()

            perturbed_dataset = perturb_dataset_interactions(
                dataset,
                strength=perturbation_cfg.get("strength", 0.1),
                seed=perturbation_cfg.get("seed", 42),
            )

            if causal_mode == "cross_view":
                # Build two independent views for causal agreement:
                #   - sequence view (default: dual_frozen_backbone)
                #   - graph view    (default: mixhop_propagation)
                seq_cfg = causal_cfg.get("sequence_model", {"name": "dual_frozen_backbone"})
                graph_cfg = causal_cfg.get("graph_model", {"name": "mixhop_propagation"})

                seq_predictor = create_predictor(seq_cfg)
                graph_predictor = create_predictor(graph_cfg)

                p_seq = seq_predictor.fit_predict(dataset, train_mask=train_mask)
                p_graph = graph_predictor.fit_predict(dataset, train_mask=train_mask)
                p_seq_pert = seq_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)
                p_graph_pert = graph_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)

                cross_view = compute_cross_view_causal_metrics(
                    p_seq=p_seq,
                    p_graph=p_graph,
                    p_seq_pert=p_seq_pert,
                    p_graph_pert=p_graph_pert,
                    weight=causal_weight if causal_weight > 0 else 1.0,
                )
                summary_payload["causal"] = {
                    "mode": "cross_view",
                    "sequence_model": str(seq_cfg.get("name", "dual_frozen_backbone")),
                    "graph_model": str(graph_cfg.get("name", "mixhop_propagation")),
                    "metrics": cross_view,
                }
                summary_payload["causal_score"] = cross_view["causal_score"]
            elif causal_mode == "mas":
                # ---------------------------------------------------------------
                # Pillar 3: Masked AutoEncoder Self-Supervision (MAS)
                # ---------------------------------------------------------------
                # Load frozen drug + protein embeddings, then run MASHead on each.
                # The quality of the embeddings is measured by how well masked dims
                # can be reconstructed from unmasked dims (linear decoder).
                from src.c2dti.backbones import load_frozen_entity_embeddings

                mas_cfg = causal_cfg.get("mas_config", {})
                drug_npz = mas_cfg.get("drug_npz_path")    # can be None → hash fallback
                prot_npz = mas_cfg.get("prot_npz_path")    # can be None → hash fallback
                emb_dim  = int(mas_cfg.get("embedding_dim", 768))
                mask_ratio = float(mas_cfg.get("mask_ratio", 0.15))
                mas_seed   = int(mas_cfg.get("seed", 42))

                # Resolve NPZ paths relative to repo root (same as dataset paths).
                drug_npz_resolved = str(_resolve_runtime_path(drug_npz)) if drug_npz else None
                prot_npz_resolved = str(_resolve_runtime_path(prot_npz)) if prot_npz else None

                drug_embeddings = load_frozen_entity_embeddings(
                    entities=dataset.drugs,
                    npz_path=drug_npz_resolved,
                    default_dim=emb_dim,
                )
                prot_embeddings = load_frozen_entity_embeddings(
                    entities=dataset.targets,
                    npz_path=prot_npz_resolved,
                    default_dim=emb_dim,
                )

                mas_metrics = compute_mas_losses(
                    drug_embeddings=drug_embeddings,
                    prot_embeddings=prot_embeddings,
                    mask_ratio=mask_ratio,
                    seed=mas_seed,
                    weight=causal_weight if causal_weight > 0 else 1.0,
                )
                summary_payload["causal"] = {
                    "mode": "mas",
                    "embedding_dim": emb_dim,
                    "mask_ratio": mask_ratio,
                    "n_drugs": len(dataset.drugs),
                    "n_targets": len(dataset.targets),
                    "metrics": mas_metrics,
                }
                summary_payload["causal_score"] = mas_metrics["mas_score"]
            elif causal_mode == "irm_cf":
                # ---------------------------------------------------------------
                # Pillar 4: IRM + Counterfactual Loss
                # ---------------------------------------------------------------
                # IRM part: split drugs into n_envs groups; variance of per-group
                # MSE is the IRM penalty — low variance = invariant model.
                # CF part: for each positive pair swap the target for a random one;
                # the model should score these fake pairs near 0.
                irm_cf_cfg = causal_cfg.get("irm_cf_config", {})
                n_envs        = int(irm_cf_cfg.get("n_envs", 4))
                pos_threshold = float(irm_cf_cfg.get("pos_threshold", 0.5))
                n_cf_pairs    = int(irm_cf_cfg.get("n_cf_pairs", 1000))
                irm_weight    = float(irm_cf_cfg.get("irm_weight", 1.0))
                cf_weight     = float(irm_cf_cfg.get("cf_weight", 1.0))
                irm_cf_seed   = int(irm_cf_cfg.get("seed", 42))

                # Flatten predictions and labels for all known pairs.
                flat_preds  = predictions.ravel()
                flat_labels = dataset.interactions.ravel()

                irm_cf_metrics = compute_irm_cf_losses(
                    predictions=flat_preds,
                    labels=flat_labels,
                    n_drugs=len(dataset.drugs),
                    n_targets=len(dataset.targets),
                    n_envs=n_envs,
                    pos_threshold=pos_threshold,
                    n_cf_pairs=n_cf_pairs,
                    irm_weight=irm_weight,
                    cf_weight=cf_weight,
                    seed=irm_cf_seed,
                )
                summary_payload["causal"] = {
                    "mode": "irm_cf",
                    "n_envs": n_envs,
                    "pos_threshold": pos_threshold,
                    "n_cf_pairs_requested": n_cf_pairs,
                    "metrics": irm_cf_metrics,
                }
                summary_payload["causal_score"] = irm_cf_metrics["irm_cf_score"]
            elif causal_mode == "unified":
                # ---------------------------------------------------------------
                # Phase 5: Unified 4-Pillar Causal Objective
                # ---------------------------------------------------------------
                # Runs ALL active pillars (cross_view, mas, irm_cf) in one call.
                # Each pillar has its own lambda weight; set lambda to 0 to ablate.
                # This is the full C2DTI causal objective:
                #   L_total = λ_xview·L_XVIEW + λ_mas·L_MAS + λ_irm·L_IRM + λ_cf·L_CF
                #   unified_causal_score = 1 / (1 + L_total)
                from src.c2dti.unified_scorer import UnifiedC2DTIScorer
                from src.c2dti.backbones import load_frozen_entity_embeddings

                scorer = UnifiedC2DTIScorer(causal_cfg)

                # --- Pillar 2 inputs: two-view predictions ---
                seq_cfg   = causal_cfg.get("sequence_model", {"name": "dual_frozen_backbone"})
                graph_cfg = causal_cfg.get("graph_model",    {"name": "mixhop_propagation"})
                seq_predictor   = create_predictor(seq_cfg)
                graph_predictor = create_predictor(graph_cfg)
                p_seq        = seq_predictor.fit_predict(dataset, train_mask=train_mask)
                p_graph      = graph_predictor.fit_predict(dataset, train_mask=train_mask)
                p_seq_pert   = seq_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)
                p_graph_pert = graph_predictor.fit_predict(perturbed_dataset, train_mask=train_mask)

                # --- Pillar 3 inputs: frozen embeddings ---
                mas_cfg = causal_cfg.get("mas_config", {})
                emb_dim = int(mas_cfg.get("embedding_dim", 768))
                drug_npz = mas_cfg.get("drug_npz_path")
                prot_npz = mas_cfg.get("prot_npz_path")
                drug_npz_resolved = str(_resolve_runtime_path(drug_npz)) if drug_npz else None
                prot_npz_resolved = str(_resolve_runtime_path(prot_npz)) if prot_npz else None
                drug_embeddings = load_frozen_entity_embeddings(
                    dataset.drugs, npz_path=drug_npz_resolved, default_dim=emb_dim
                )
                prot_embeddings = load_frozen_entity_embeddings(
                    dataset.targets, npz_path=prot_npz_resolved, default_dim=emb_dim
                )

                unified = scorer.score(
                    predictions=predictions,
                    labels=dataset.interactions,
                    n_drugs=len(dataset.drugs),
                    n_targets=len(dataset.targets),
                    seq_predictions=p_seq,
                    graph_predictions=p_graph,
                    seq_predictions_pert=p_seq_pert,
                    graph_predictions_pert=p_graph_pert,
                    drug_embeddings=drug_embeddings,
                    prot_embeddings=prot_embeddings,
                )
                summary_payload["causal"] = {
                    "mode": "unified",
                    "lambdas": {
                        "xview": scorer.lambda_xview,
                        "mas":   scorer.lambda_mas,
                        "irm":   scorer.lambda_irm,
                        "cf":    scorer.lambda_cf,
                    },
                    "metrics": unified,
                }
                summary_payload["causal_score"] = unified["unified_causal_score"]
            else:
                perturbed_predictions = predictor.fit_predict(perturbed_dataset, train_mask=train_mask)
                reliability = compute_causal_reliability_score(
                    baseline_predictions=predictions,
                    perturbed_predictions=perturbed_predictions,
                    weight=causal_weight if causal_weight > 0 else 1.0,
                )
                summary_payload["causal"] = {
                    "mode": "reliability",
                    "metric": "prediction_stability",
                }
                summary_payload["causal_score"] = reliability
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
