from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_data_sources


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    role: str
    loader: str
    url: str
    raw_dir: Path
    expected_extensions: tuple[str, ...]
    description: str
    citation: str
    measured_variables: tuple[str, ...]


class DataRegistry:
    def __init__(self, sources_path: str | Path | None = None) -> None:
        self.sources = load_data_sources(sources_path)

    def names(self) -> list[str]:
        return list(self.sources.keys())

    def get(self, dataset_name: str) -> DatasetInfo:
        if dataset_name not in self.sources:
            known = ", ".join(self.names())
            raise KeyError(f"Unknown dataset '{dataset_name}'. Known datasets: {known}")
        raw = self.sources[dataset_name]
        raw_dir = PROJECT_ROOT / raw["raw_dir"]
        expected_extensions = _as_string_tuple(raw.get("expected_extensions", []))
        measured_variables = _as_string_tuple(raw.get("measured_variables", []))
        if not measured_variables:
            measured_variables = _default_measured_variables(dataset_name)
        return DatasetInfo(
            name=dataset_name,
            role=raw.get("role", ""),
            loader=raw.get("loader", dataset_name),
            url=raw.get("url", ""),
            raw_dir=raw_dir,
            expected_extensions=tuple(x.lower() for x in expected_extensions),
            description=raw.get("description", ""),
            citation=raw.get("citation", ""),
            measured_variables=measured_variables,
        )

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.names():
            info = self.get(name)
            rows.append(
                {
                    "dataset": info.name,
                    "role": info.role,
                    "url": info.url,
                    "raw_dir": str(info.raw_dir),
                    "loader": info.loader,
                    "description": info.description,
                    "citation": info.citation,
                    "measured_variables": "; ".join(info.measured_variables),
                }
            )
        return rows


def list_available_raw_files(dataset_name: str) -> list[Path]:
    registry = DataRegistry()
    info = registry.get(dataset_name)
    info.raw_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for path in info.raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in info.expected_extensions and path.name != ".gitkeep" and path.stat().st_size > 0:
            files.append(path)
    return sorted(files)


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return tuple(part.strip().strip("'\"") for part in text.split(",") if part.strip())
    if isinstance(value, dict):
        return tuple(str(v) for v in value.values())
    return tuple(str(x) for x in value)


def _default_measured_variables(dataset_name: str) -> tuple[str, ...]:
    defaults = {
        "sonderborg": ("heat load", "feed/supply temperature", "return temperature", "plant-level operating profiles"),
        "flensburg": ("heat load", "feed/supply temperature"),
        "xai4heat": ("sparse measured substation nodes", "substation temperatures", "energy transmission", "outdoor temperature if available"),
        "aalborg": ("consumer heat demand profiles",),
    }
    return defaults.get(dataset_name, tuple())


def check_dataset_available(dataset_name: str) -> bool:
    return len(list_available_raw_files(dataset_name)) > 0


def processed_file_exists(dataset_name: str) -> bool:
    return (PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_processed.csv").exists()


def data_availability_rows(notes: dict[str, str] | None = None, download_status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    notes = notes or {}
    download_status = download_status or {}
    registry = DataRegistry()
    rows: list[dict[str, Any]] = []
    for name in registry.names():
        info = registry.get(name)
        raw_files = list_available_raw_files(name)
        rows.append(
            {
                "dataset_name": name,
                "role": info.role,
                "source_url": info.url,
                "available": bool(raw_files),
                "raw_file_count": len(raw_files),
                "processed_file_exists": processed_file_exists(name),
                "download_status": download_status.get(name, "not_attempted"),
                "note": notes.get(name, "available locally" if raw_files else f"manual download needed from {info.url}"),
            }
        )
    return rows
