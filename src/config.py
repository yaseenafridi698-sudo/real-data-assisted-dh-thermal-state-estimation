from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback is for lean runtime environments.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            data = yaml.safe_load(f) or {}
        else:
            data = _minimal_yaml_load(f.read())
    return data


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser for the local quality-gate runtime.

    The repository normally uses PyYAML. Some Codex desktop runtimes do not ship
    optional YAML dependencies, so this keeps simple nested config files readable
    for asset generation and quality checks without changing scientific results.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip().strip("'\"")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except Exception:
        return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "default_config.yaml"
    cfg = load_yaml(cfg_path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def load_data_sources(path: str | Path | None = None) -> dict[str, Any]:
    src_path = Path(path) if path else PROJECT_ROOT / "config" / "data_sources.yaml"
    return load_yaml(src_path)


def ensure_project_dirs() -> None:
    for rel in [
        "data/raw",
        "data/processed",
        "data/external",
        "results",
        "figures",
        "paper/figures",
        "paper/tables",
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)
