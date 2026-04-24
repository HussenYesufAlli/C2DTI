from pathlib import Path
from typing import Dict, List, Any

from src.c2dti.causal_objective import validate_causal_config


def _validate_dataset_config(dataset_cfg: Any) -> List[str]:
    """Validate optional dataset configuration for dataset-backed pipeline runs."""
    if dataset_cfg is None:
        return []

    if not isinstance(dataset_cfg, dict):
        return ["dataset config must be a mapping"]

    errors: List[str] = []
    dataset_name = dataset_cfg.get("name")
    if not dataset_name:
        errors.append("dataset.name is required when dataset config is provided")
    elif str(dataset_name).upper() not in {"BINDINGDB", "BINDINGDB_KD", "DAVIS", "KIBA"}:
        errors.append("dataset.name must be one of: BindingDB, BindingDB_Kd, DAVIS, KIBA")

    if not dataset_cfg.get("path"):
        errors.append("dataset.path is required when dataset config is provided")

    allow_placeholder = dataset_cfg.get("allow_placeholder", True)
    if not isinstance(allow_placeholder, bool):
        errors.append("dataset.allow_placeholder must be a boolean")

    return errors


def _validate_model_config(model_cfg: Any) -> List[str]:
    """Validate optional model configuration used by the dataset-backed pipeline path.

        Supported model names and their config keys:
            simple_baseline           : no additional keys required
            dual_frozen_backbone      : chemberta_npz_path (str, optional), ankh_npz_path (str, optional),
                                                                    fusion_alpha (float in [0, 1]), max_calibration_samples (int >= 1000), seed (int)
            end_to_end_char_encoder   : embedding_dim (int), epochs (int), lr (float),
                                                                    max_drug_len (int), max_target_len (int), l2 (float), seed (int)
            matrix_factorization      : latent_dim (int), epochs (int), lr (float), seed (int)
            mixhop_propagation        : top_k (int), hop_weights (list[float])
            interaction_cross_attention : latent_dim (int), epochs (int), lr (float),
                                                                        seed (int), attention_temperature (float), top_k (int)
            shared objective selector : objective ("auto" | "binary_classification" | "regression")
    """
    if model_cfg is None:
        return []

    if not isinstance(model_cfg, dict):
        return ["model config must be a mapping"]

    model_name = model_cfg.get("name", "simple_baseline")
    if not isinstance(model_name, str):
        return ["model.name must be a string"]

    valid_names = {
        "simple_baseline",
        "dual_frozen_backbone",
        "end_to_end_char_encoder",
        "matrix_factorization",
        "mixhop_propagation",
        "interaction_cross_attention",
    }
    if model_name.strip().lower() not in valid_names:
        return [f"model.name must be one of: {', '.join(sorted(valid_names))}"]

    errors: List[str] = []

    if "latent_dim" in model_cfg:
        val = model_cfg["latent_dim"]
        if not isinstance(val, int) or val < 1:
            errors.append("model.latent_dim must be a positive integer")

    if "epochs" in model_cfg:
        val = model_cfg["epochs"]
        if not isinstance(val, int) or val < 1:
            errors.append("model.epochs must be a positive integer")

    if "lr" in model_cfg:
        val = model_cfg["lr"]
        if not isinstance(val, (int, float)) or float(val) <= 0.0:
            errors.append("model.lr must be a positive number")

    if "seed" in model_cfg:
        val = model_cfg["seed"]
        if not isinstance(val, int):
            errors.append("model.seed must be an integer")

    if "attention_temperature" in model_cfg:
        val = model_cfg["attention_temperature"]
        if not isinstance(val, (int, float)) or float(val) <= 0.0:
            errors.append("model.attention_temperature must be a positive number")

    if "top_k" in model_cfg:
        val = model_cfg["top_k"]
        if not isinstance(val, int) or val < 1:
            errors.append("model.top_k must be a positive integer")

    if "hop_weights" in model_cfg:
        val = model_cfg["hop_weights"]
        if not isinstance(val, list) or len(val) == 0:
            errors.append("model.hop_weights must be a non-empty list of numbers")
        elif any((not isinstance(v, (int, float)) or float(v) < 0.0) for v in val):
            errors.append("model.hop_weights entries must be non-negative numbers")

    if "objective" in model_cfg:
        val = str(model_cfg["objective"]).strip().lower()
        valid_objectives = {
            "auto",
            "binary",
            "binary_classification",
            "classification",
            "regression",
            "continuous",
        }
        if val not in valid_objectives:
            errors.append(
                "model.objective must be one of: auto, binary_classification, regression"
            )

    if "chemberta_npz_path" in model_cfg:
        val = model_cfg["chemberta_npz_path"]
        if val is not None and not isinstance(val, str):
            errors.append("model.chemberta_npz_path must be a string path or null")

    if "ankh_npz_path" in model_cfg:
        val = model_cfg["ankh_npz_path"]
        if val is not None and not isinstance(val, str):
            errors.append("model.ankh_npz_path must be a string path or null")

    if "fusion_alpha" in model_cfg:
        val = model_cfg["fusion_alpha"]
        if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
            errors.append("model.fusion_alpha must be a number in [0, 1]")

    if "max_calibration_samples" in model_cfg:
        val = model_cfg["max_calibration_samples"]
        if not isinstance(val, int) or val < 1000:
            errors.append("model.max_calibration_samples must be an integer >= 1000")

    if "embedding_dim" in model_cfg:
        val = model_cfg["embedding_dim"]
        if not isinstance(val, int) or val < 8:
            errors.append("model.embedding_dim must be an integer >= 8")

    if "max_drug_len" in model_cfg:
        val = model_cfg["max_drug_len"]
        if not isinstance(val, int) or val < 8:
            errors.append("model.max_drug_len must be an integer >= 8")

    if "max_target_len" in model_cfg:
        val = model_cfg["max_target_len"]
        if not isinstance(val, int) or val < 8:
            errors.append("model.max_target_len must be an integer >= 8")

    if "l2" in model_cfg:
        val = model_cfg["l2"]
        if not isinstance(val, (int, float)) or float(val) < 0.0:
            errors.append("model.l2 must be a non-negative number")

    return errors


def _validate_perturbation_config(perturbation_cfg: Any) -> List[str]:
    """Validate optional perturbation settings used for causal reliability."""
    if perturbation_cfg is None:
        return []

    if not isinstance(perturbation_cfg, dict):
        return ["perturbation config must be a mapping"]

    errors: List[str] = []
    strength = perturbation_cfg.get("strength", 0.1)
    if not isinstance(strength, (int, float)) or strength < 0.0 or strength > 1.0:
        errors.append("perturbation.strength must be a number between 0 and 1")

    seed = perturbation_cfg.get("seed", 42)
    if not isinstance(seed, int):
        errors.append("perturbation.seed must be an integer")

    return errors


def _validate_split_config(split_cfg: Any) -> List[str]:
    """Validate optional split configuration for train/test evaluation.

    Supported keys:
      strategy   : one of "random", "cold_drug", "cold_target"  (default: "random")
      test_ratio : float between 0 and 1 exclusive              (default: 0.2)
      seed       : integer for reproducibility                   (default: 42)
    """
    if split_cfg is None:
        return []

    if not isinstance(split_cfg, dict):
        return ["split config must be a mapping"]

    errors: List[str] = []

    valid_strategies = {"random", "cold_drug", "cold_target"}
    strategy = split_cfg.get("strategy", "random")
    if not isinstance(strategy, str) or strategy.strip().lower() not in valid_strategies:
        errors.append(
            f"split.strategy must be one of: {', '.join(sorted(valid_strategies))}"
        )

    test_ratio = split_cfg.get("test_ratio", 0.2)
    if not isinstance(test_ratio, (int, float)) or not (0.0 < float(test_ratio) < 1.0):
        errors.append("split.test_ratio must be a number strictly between 0 and 1")

    seed = split_cfg.get("seed", 42)
    if not isinstance(seed, int):
        errors.append("split.seed must be an integer")

    return errors


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(cfg, dict):
        return ["Config root must be a mapping."]

    if not cfg.get("name"):
        errors.append("Missing required key: name")

    if not cfg.get("protocol"):
        errors.append("Missing required key: protocol")

    output = cfg.get("output", {})
    if not isinstance(output, dict) or not output.get("base_dir"):
        errors.append("Missing required key: output.base_dir")

    causal_cfg = cfg.get("causal")
    causal_errors = validate_causal_config(causal_cfg)
    errors.extend(causal_errors)

    errors.extend(_validate_dataset_config(cfg.get("dataset")))
    errors.extend(_validate_model_config(cfg.get("model")))
    errors.extend(_validate_perturbation_config(cfg.get("perturbation")))
    errors.extend(_validate_split_config(cfg.get("split")))

    return errors
