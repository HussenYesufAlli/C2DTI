from pathlib import Path
from typing import Dict, List, Any

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

    return errors
