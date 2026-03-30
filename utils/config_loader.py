from pathlib import Path

import yaml


def resolve_config_path(config_path: str, base_dir: Path) -> Path:
    path = Path(config_path)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (base_dir / path).resolve()


def load_config(config_path: str, base_dir: Path) -> dict:
    with open(resolve_config_path(config_path, base_dir), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
