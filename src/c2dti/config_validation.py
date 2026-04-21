from pathlib import Path
from typing import Dict, List, Any

from src.c2dti.causal_objective import validate_causal_config


def _validate_dataset_config(dataset_cfg: Any) -> List[str]:
    """Validate optional dataset configuration for real pipeline runs."""
    if dataset_cfg is None:
        return []

    if not isinstance(dataset_cfg, dict):
        return ["dataset config must be a mapping"]

    errors: List[str] = []
    dataset_name = dataset_cfg.get("name")
    if not dataset_name:
        errors.append("dataset.name is required when dataset config is provided")
    elif str(dataset_name).upper() not in {"BINDINGDB", "DAVIS", "KIBA"}:
        errors.append("dataset.name must be one of: BindingDB, DAVIS, KIBA")

    if not dataset_cfg.get("path"):
        errors.append("dataset.path is required when dataset config is provided")

    return errors


def _validate_model_config(model_cfg: Any) -> List[str]:
    """Validate optional model configuration used by the real pipeline path."""
    if model_cfg is None:
        return []

    if not isinstance(model_cfg, dict):
        return ["model config must be a mapping"]

    model_name = model_cfg.get("name", "simple_baseline")
    if not isinstance(model_name, str):
        return ["model.name must be a string"]

    if model_name.strip().lower() not in {"simple_baseline"}:
        return ["model.name must be one of: simple_baseline"]

    return []


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

    # Validate optional causal configuration
    causal_cfg = cfg.get("causal")
    causal_errors = validate_causal_config(causal_cfg)
    errors.extend(causal_errors)

    errors.extend(_validate_dataset_config(cfg.get("dataset")))
    errors.extend(_validate_model_config(cfg.get("model")))
    errors.extend(_validate_perturbation_config(cfg.get("perturbation")))

    return errors
