from pathlib import Path
from typing import Dict, List, Any

from src.c2dti.causal_objective import validate_causal_config

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

    return errors
