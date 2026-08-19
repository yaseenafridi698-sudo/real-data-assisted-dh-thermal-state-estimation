from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import requests
import pandas as pd

from .config import PROJECT_ROOT
from .data_registry import DataRegistry, data_availability_rows, list_available_raw_files
from .utils import ensure_dir, manual_download_message


def _record_id_from_zenodo_url(url: str) -> str | None:
    match = re.search(r"zenodo\.org/(?:records|record)/(\d+)", url)
    return match.group(1) if match else None


def _download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    ensure_dir(dest.parent)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def _maybe_extract_zip(path: Path, out_dir: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    try:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        print(f"Downloaded file is not a valid zip archive: {path}")


def _download_zenodo_record(record_id: str, out_dir: Path) -> list[Path]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    files = payload.get("files", [])
    downloaded: list[Path] = []
    for item in files:
        links = item.get("links", {})
        file_url = links.get("self") or links.get("download")
        filename = item.get("key") or item.get("filename") or Path(file_url).name
        if not file_url:
            continue
        dest = out_dir / filename
        if dest.exists():
            downloaded.append(dest)
            continue
        print(f"Downloading {filename} from Zenodo record {record_id}")
        downloaded_path = _download_file(file_url, dest)
        _maybe_extract_zip(downloaded_path, out_dir)
        downloaded.append(downloaded_path)
    return downloaded


def download_dataset(dataset_name: str) -> tuple[bool, str, str]:
    registry = DataRegistry()
    info = registry.get(dataset_name)
    ensure_dir(info.raw_dir)
    existing = list_available_raw_files(dataset_name)
    if existing:
        print(f"{dataset_name}: raw directory already contains {len(existing)} file(s); leaving them untouched.")
        return True, "already_available", f"{len(existing)} raw file(s) already present"

    url = info.url
    try:
        record_id = _record_id_from_zenodo_url(url)
        if record_id:
            downloaded = _download_zenodo_record(record_id, info.raw_dir)
            if downloaded:
                print(f"{dataset_name}: downloaded {len(downloaded)} file(s).")
                return True, "downloaded", f"downloaded {len(downloaded)} file(s) via Zenodo API"
        if "mendeley.com" in url:
            print(f"{dataset_name}: Mendeley downloads often require browser or API-specific access.")
            print(manual_download_message(dataset_name, url))
            return False, "manual_required", manual_download_message(dataset_name, url)
        if url:
            filename = Path(url.rstrip("/")).name or f"{dataset_name}_download"
            dest = info.raw_dir / filename
            downloaded_path = _download_file(url, dest)
            _maybe_extract_zip(downloaded_path, info.raw_dir)
            print(f"{dataset_name}: downloaded {downloaded_path.name}.")
            return True, "downloaded", f"downloaded {downloaded_path.name}"
    except Exception as exc:
        print(f"{dataset_name}: automatic download failed: {exc}")
        print(manual_download_message(dataset_name, url))
        return False, "failed", f"{exc}. {manual_download_message(dataset_name, url)}"

    print(manual_download_message(dataset_name, url))
    return False, "manual_required", manual_download_message(dataset_name, url)


def download_all_datasets() -> dict[str, bool]:
    registry = DataRegistry()
    statuses: dict[str, bool] = {}
    notes: dict[str, str] = {}
    status_text: dict[str, str] = {}
    for name in registry.names():
        ok, status, note = download_dataset(name)
        statuses[name] = ok
        status_text[name] = status
        notes[name] = note
    write_data_availability_report(notes, status_text)
    return statuses


def write_data_availability_report(notes: dict[str, str] | None = None, download_status: dict[str, str] | None = None) -> Path:
    ensure_dir(PROJECT_ROOT / "results")
    rows = data_availability_rows(notes=notes, download_status=download_status)
    path = PROJECT_ROOT / "results" / "data_availability_report.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    if not any(row["available"] for row in rows):
        warning = PROJECT_ROOT / "results" / "DATA_WARNING_REAL_DATA_NOT_FOUND.txt"
        warning.write_text(
            "No configured real district-heating datasets were found in data/raw/.\n"
            "Quick demo may use fallback synthetic-realistic data only for software testing.\n"
            "Journal results require the real datasets listed in config/data_sources.yaml.\n",
            encoding="utf-8",
        )
    return path
