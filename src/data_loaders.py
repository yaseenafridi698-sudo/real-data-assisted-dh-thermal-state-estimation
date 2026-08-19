from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_yaml
from .data_registry import DataRegistry
from .utils import ensure_dir


TIME_KEYWORDS = ["timestamp", "datetime", "date_time", "time", "date", "utc", "local"]
HEAT_KEYWORDS = ["heat", "load", "demand", "power", "energy", "q", "mw", "kw", "output"]
SUPPLY_KEYWORDS = ["supply", "feed", "forward", "fremlob", "fremloeb", "fremløb", "flow_temperature", "t_supply", "temp_supply"]
RETURN_KEYWORDS = ["return", "retur", "back", "back_flow", "t_return", "temp_return"]
AMBIENT_KEYWORDS = ["outdoor", "ambient", "weather", "air_temp", "t_out", "outside"]
ID_KEYWORDS = ["plant", "station", "substation", "meter", "building", "consumer", "id"]


def normalize_column_name(col: object) -> str:
    text = str(col).strip().lower()
    text = text.replace("\u00f8", "o").replace("\u00e5", "a").replace("\u00e6", "ae")
    text = text.replace("ø", "o").replace("å", "a").replace("æ", "ae")
    text = text.replace("Ã¸", "o").replace("Ã¥", "a").replace("Ã¦", "ae")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def _candidate_files(raw_dir: Path) -> list[Path]:
    exts = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}
    for archive in raw_dir.rglob("*.zip"):
        extract_dir = archive.with_suffix("")
        if extract_dir.exists():
            continue
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_dir)
        except Exception as exc:
            print(f"Could not extract {archive}: {exc}")
    return sorted([p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def _is_numeric_like(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() > 0.5


def _score_column(name: str, keywords: Iterable[str], exclude: Iterable[str] = ()) -> int:
    score = 0
    low = name.lower()
    for word in keywords:
        if word in low:
            score += 2 if low == word else 1
    for word in exclude:
        if word in low:
            score -= 2
    return score


def _load_user_mapping(dataset_name: str) -> dict[str, str]:
    path = PROJECT_ROOT / "config" / "user_column_mapping.yaml"
    if not path.exists():
        return {}
    data = load_yaml(path)
    raw = data.get(dataset_name, {}) or {}
    return {str(k): normalize_column_name(v) for k, v in raw.items() if v not in (None, "")}


def _mapped_column(df: pd.DataFrame, mapping: dict[str, str], logical_name: str) -> str | None:
    col = mapping.get(logical_name)
    if col and col in df.columns:
        return col
    return None


def _find_timestamp_column(df: pd.DataFrame) -> tuple[str | None, int]:
    names = list(df.columns)
    scored = sorted(names, key=lambda c: _score_column(c, TIME_KEYWORDS), reverse=True)
    for col in scored:
        score = _score_column(col, TIME_KEYWORDS)
        if score <= 0:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().mean() > 0.45:
            return col, score
    for col in names:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().mean() > 0.75:
            return col, 1
    return None, 0


def _find_numeric_column(df: pd.DataFrame, keywords: Iterable[str], exclude: Iterable[str] = ()) -> tuple[str | None, int]:
    candidates: list[tuple[int, str]] = []
    for col in df.columns:
        if not _is_numeric_like(df[col]):
            continue
        score = _score_column(col, keywords, exclude=exclude)
        if score > 0:
            candidates.append((score, col))
    if not candidates:
        return None, 0
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][0]


def _find_id_column(df: pd.DataFrame) -> tuple[str | None, int]:
    candidates: list[tuple[int, str]] = []
    for col in df.columns:
        score = _score_column(col, ID_KEYWORDS)
        if score > 0 and df[col].nunique(dropna=True) <= max(1000, len(df) // 3):
            candidates.append((score, col))
    if not candidates:
        return None, 0
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][0]


def _write_column_report(
    dataset_name: str,
    frames: list[pd.DataFrame],
    files: list[Path],
    detections: list[dict[str, object]] | None = None,
) -> None:
    rows = []
    for file, df in zip(files, frames):
        for col in df.columns:
            series = df[col]
            rows.append(
                {
                    "dataset": dataset_name,
                    "file": str(file),
                    "column": col,
                    "non_null_fraction": float(series.notna().mean()),
                    "numeric_like": _is_numeric_like(series),
                    "timestamp_score": _score_column(col, TIME_KEYWORDS),
                    "heat_score": _score_column(col, HEAT_KEYWORDS, exclude=["temp", "temperature", "supply", "return", "ambient"]),
                    "supply_score": _score_column(col, SUPPLY_KEYWORDS),
                    "return_score": _score_column(col, RETURN_KEYWORDS),
                    "ambient_score": _score_column(col, AMBIENT_KEYWORDS),
                    "id_score": _score_column(col, ID_KEYWORDS),
                    "sample_values": "; ".join(map(str, series.dropna().head(3).tolist())),
                }
            )
    ensure_dir(PROJECT_ROOT / "results")
    pd.DataFrame(rows).to_csv(PROJECT_ROOT / "results" / f"{dataset_name}_column_detection_report.csv", index=False)
    if detections is not None:
        pd.DataFrame(detections).to_csv(PROJECT_ROOT / "results" / f"{dataset_name}_detected_columns.csv", index=False)


def _standardize_frame(df: pd.DataFrame, dataset_name: str, path: Path) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    mapping = _load_user_mapping(dataset_name)

    timestamp_col = _mapped_column(df, mapping, "timestamp")
    timestamp_col, timestamp_score = (timestamp_col, 99) if timestamp_col else _find_timestamp_column(df)

    heat_col = _mapped_column(df, mapping, "heat_load_kw")
    heat_col, heat_score = (
        (heat_col, 99)
        if heat_col
        else _find_numeric_column(df, HEAT_KEYWORDS, exclude=["temp", "temperature", "supply", "return", "ambient"])
    )

    supply_col = _mapped_column(df, mapping, "supply_temp_C")
    supply_col, supply_score = (supply_col, 99) if supply_col else _find_numeric_column(df, SUPPLY_KEYWORDS)

    return_col = _mapped_column(df, mapping, "return_temp_C")
    return_col, return_score = (return_col, 99) if return_col else _find_numeric_column(df, RETURN_KEYWORDS)

    ambient_col = _mapped_column(df, mapping, "ambient_temp_C")
    ambient_col, ambient_score = (ambient_col, 99) if ambient_col else _find_numeric_column(df, AMBIENT_KEYWORDS)

    id_col = _mapped_column(df, mapping, "substation_id") or _mapped_column(df, mapping, "plant_id")
    id_col, id_score = (id_col, 99) if id_col else _find_id_column(df)

    missing = []
    if timestamp_col is None:
        missing.append("timestamp")
    if heat_col is None and dataset_name != "xai4heat":
        missing.append("heat_load")
    if supply_col is None:
        missing.append("supply_temp")
    if missing:
        raise ValueError(
            f"{path.name}: could not identify required columns: {', '.join(missing)}. "
            "Review results/<dataset>_column_detection_report.csv and edit config/user_column_mapping.yaml."
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    out["heat_load_kw"] = pd.to_numeric(df[heat_col], errors="coerce") if heat_col else np.nan
    out["supply_temp_C"] = pd.to_numeric(df[supply_col], errors="coerce") if supply_col else np.nan
    out["return_temp_C"] = pd.to_numeric(df[return_col], errors="coerce") if return_col else np.nan
    out["ambient_temp_C"] = pd.to_numeric(df[ambient_col], errors="coerce") if ambient_col else np.nan
    out["source_dataset"] = dataset_name
    if id_col is not None:
        id_name = "substation_id" if dataset_name == "xai4heat" else "plant_id"
        out[id_name] = df[id_col].astype(str).fillna("unknown")
    if "plant_id" not in out.columns:
        out["plant_id"] = np.nan
    if "substation_id" not in out.columns:
        out["substation_id"] = np.nan
    if return_col is None and dataset_name == "flensburg":
        out["return_temp_C"] = 50.0
        out["return_temp_assumed"] = True
    elif "return_temp_assumed" not in out.columns:
        out["return_temp_assumed"] = False
    out.attrs["column_detection"] = {
        "dataset": dataset_name,
        "file": str(path),
        "timestamp": timestamp_col,
        "timestamp_score": timestamp_score,
        "heat_load_kw": heat_col,
        "heat_score": heat_score,
        "supply_temp_C": supply_col,
        "supply_score": supply_score,
        "return_temp_C": return_col,
        "return_score": return_score,
        "ambient_temp_C": ambient_col,
        "ambient_score": ambient_score,
        "id_col": id_col,
        "id_score": id_score,
        "mapping_file_used": bool(mapping),
        "uncertain": bool(min(timestamp_score, supply_score, heat_score if dataset_name != "xai4heat" else 99) < 1),
    }
    return out.dropna(subset=["timestamp"])


def _load_generic_dataset(dataset_name: str, raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = _candidate_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV/XLSX/TXT files found in {raw_dir}. "
            f"Please manually download the dataset and place files in data/raw/{dataset_name}/."
        )
    raw_frames: list[pd.DataFrame] = []
    standardized: list[pd.DataFrame] = []
    errors: list[str] = []
    used_files: list[Path] = []
    detections: list[dict[str, object]] = []
    for file in files:
        df = _read_table(file)
        if df is None or df.empty:
            continue
        normalized = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
        raw_frames.append(normalized)
        used_files.append(file)
        try:
            standard = _standardize_frame(df, dataset_name, file)
            detections.append(standard.attrs.get("column_detection", {"dataset": dataset_name, "file": str(file)}))
            standardized.append(standard)
        except Exception as exc:
            errors.append(str(exc))
    if raw_frames:
        _write_column_report(dataset_name, raw_frames, used_files, detections=detections)
    if not standardized:
        report = PROJECT_ROOT / "results" / f"{dataset_name}_column_detection_report.csv"
        raise ValueError(
            f"Could not automatically identify columns for {dataset_name}. "
            f"A column report was saved to {report}. Edit config/user_column_mapping.yaml if needed. "
            f"Errors: {' | '.join(errors[:5])}"
        )
    out = pd.concat(standardized, ignore_index=True, sort=False)
    return out.sort_values("timestamp")


def load_sonderborg(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("sfjv_dhs_data_*.csv"))
    if csv_files:
        frames = []
        detections = []
        raw_frames = []
        for file in csv_files:
            df = _read_table(file)
            if df is None or df.empty:
                continue
            df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
            raw_frames.append(df)
            time_col, _ = _find_timestamp_column(df)
            if time_col is None:
                continue
            heat_cols = [c for c in df.columns if re.search(r"plant\d+_heat_load", c)]
            feed_cols = [c for c in df.columns if re.search(r"plant\d+_temp_feed_flow", c)]
            back_cols = [c for c in df.columns if re.search(r"plant\d+_temp_back_flow", c)]
            if not heat_cols or not feed_cols:
                continue
            heat = df[heat_cols].apply(pd.to_numeric, errors="coerce")
            feed = df[feed_cols].apply(pd.to_numeric, errors="coerce")
            back = df[back_cols].apply(pd.to_numeric, errors="coerce") if back_cols else pd.DataFrame(index=df.index)
            weights = heat.where(heat > 0)
            supply = (feed.to_numpy(dtype=float) * weights.to_numpy(dtype=float)).sum(axis=1) / np.maximum(weights.sum(axis=1).to_numpy(dtype=float), 1e-9)
            if not back.empty:
                return_temp = (back.to_numpy(dtype=float) * weights.iloc[:, : back.shape[1]].to_numpy(dtype=float)).sum(axis=1) / np.maximum(weights.iloc[:, : back.shape[1]].sum(axis=1).to_numpy(dtype=float), 1e-9)
            else:
                return_temp = np.full(len(df), np.nan)
            out = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(df[time_col], errors="coerce", utc=True),
                    "heat_load_kw": heat.sum(axis=1),
                    "supply_temp_C": supply,
                    "return_temp_C": return_temp,
                    "ambient_temp_C": np.nan,
                    "source_dataset": "sonderborg",
                    "plant_id": "all_plants",
                    "substation_id": np.nan,
                    "return_temp_assumed": False,
                }
            )
            detections.append(
                {
                    "dataset": "sonderborg",
                    "file": str(file),
                    "timestamp": time_col,
                    "heat_load_kw": ";".join(heat_cols),
                    "supply_temp_C": ";".join(feed_cols),
                    "return_temp_C": ";".join(back_cols),
                    "aggregation": "sum heat load; heat-load-weighted supply/return temperature",
                }
            )
            frames.append(out.dropna(subset=["timestamp"]))
        if frames:
            _write_column_report("sonderborg", raw_frames, csv_files, detections=detections)
            return pd.concat(frames, ignore_index=True).sort_values("timestamp")
    return _load_generic_dataset("sonderborg", raw_dir)


def load_flensburg(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    xlsx_files = sorted(raw_dir.rglob("*.xlsx")) + sorted(raw_dir.rglob("*.xls"))
    for file in xlsx_files:
        df = _read_table(file)
        if df is None or df.empty:
            continue
        df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
        time_col, _ = _find_timestamp_column(df)
        if time_col is None:
            time_col = df.columns[0]
        heat_col = None
        supply_col = None
        for col in df.columns:
            if col == time_col:
                continue
            samples = " ".join(str(v) for v in df[col].dropna().head(5).tolist())
            label = normalize_column_name(samples)
            numeric_fraction = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if numeric_fraction < 0.5:
                continue
            if heat_col is None and any(token in label for token in ["warmeleistung", "waermeleistung", "heat", "load", "power"]):
                heat_col = col
            if supply_col is None and any(token in label for token in ["vorlauf", "feed", "supply", "forward"]):
                supply_col = col
        if heat_col and supply_col:
            out = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(df[time_col], errors="coerce", utc=True),
                    "heat_load_kw": pd.to_numeric(df[heat_col], errors="coerce"),
                    "supply_temp_C": pd.to_numeric(df[supply_col], errors="coerce"),
                    "return_temp_C": 50.0,
                    "ambient_temp_C": np.nan,
                    "source_dataset": "flensburg",
                    "plant_id": "total_network",
                    "substation_id": np.nan,
                    "return_temp_assumed": True,
                }
            ).dropna(subset=["timestamp", "heat_load_kw", "supply_temp_C"])
            _write_column_report(
                "flensburg",
                [df],
                [file],
                detections=[
                    {
                        "dataset": "flensburg",
                        "file": str(file),
                        "timestamp": time_col,
                        "heat_load_kw": heat_col,
                        "supply_temp_C": supply_col,
                        "return_temp_C": "assumed_50_C",
                        "return_temp_assumed": True,
                    }
                ],
            )
            return out.sort_values("timestamp")
    df = _load_generic_dataset("flensburg", raw_dir)
    if "return_temp_assumed" not in df.columns:
        missing = df["return_temp_C"].isna()
        if missing.any():
            df.loc[missing, "return_temp_C"] = 50.0
            df["return_temp_assumed"] = missing
    return df


def load_xai4heat(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    candidates = sorted(raw_dir.rglob("xai4heat_scada_L*.csv"), key=lambda p: (len(p.parts), str(p)))
    # Download packages may contain a nested duplicate copy. Retain one file
    # for each substation identifier and preserve the original measured fields.
    files: dict[str, Path] = {}
    for path in candidates:
        match = re.search(r"_L(\d+)\.csv$", path.name, flags=re.IGNORECASE)
        if match:
            files.setdefault(f"L{match.group(1)}", path)
    if files:
        frames = []
        detections = []
        for substation_id, path in sorted(files.items(), key=lambda item: int(item[0][1:])):
            raw = pd.read_csv(path)
            raw.columns = [normalize_column_name(c) for c in raw.columns]
            required = {"datetime", "t_sup_prim", "t_ret_prim"}
            if not required.issubset(raw.columns):
                continue
            out = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(raw["datetime"], errors="coerce", utc=True),
                    # XAI4HEAT e and pe are preserved in source units. They are
                    # not relabelled as heat load without authoritative unit metadata.
                    "heat_load_kw": np.nan,
                    "supply_temp_C": pd.to_numeric(raw["t_sup_prim"], errors="coerce"),
                    "return_temp_C": pd.to_numeric(raw["t_ret_prim"], errors="coerce"),
                    "ambient_temp_C": pd.to_numeric(raw.get("t_amb"), errors="coerce"),
                    "secondary_supply_temp_C": pd.to_numeric(raw.get("t_sup_sec"), errors="coerce"),
                    "secondary_return_temp_C": pd.to_numeric(raw.get("t_ret_sec"), errors="coerce"),
                    "energy_metric_e_source_units": pd.to_numeric(raw.get("e"), errors="coerce"),
                    "power_energy_metric_pe_source_units": pd.to_numeric(raw.get("pe"), errors="coerce"),
                    "reference_temp_C": pd.to_numeric(raw.get("t_ref"), errors="coerce"),
                    "source_dataset": "xai4heat",
                    "plant_id": np.nan,
                    "substation_id": substation_id,
                    "ordered_position": float(substation_id[1:]),
                    "return_temp_assumed": False,
                }
            )
            for column in ["supply_temp_C", "return_temp_C", "secondary_supply_temp_C", "secondary_return_temp_C"]:
                out.loc[(out[column] < 5.0) | (out[column] > 120.0), column] = np.nan
            out.loc[(out["ambient_temp_C"] < -35.0) | (out["ambient_temp_C"] > 45.0), "ambient_temp_C"] = np.nan
            frames.append(out.dropna(subset=["timestamp"]))
            detections.append(
                {
                    "dataset": "xai4heat",
                    "file": str(path),
                    "timestamp": "datetime",
                    "supply_temp_C": "t_sup_prim",
                    "return_temp_C": "t_ret_prim",
                    "ambient_temp_C": "t_amb",
                    "substation_id": substation_id,
                    "energy_fields": "e; pe (preserved in source units)",
                }
            )
        if frames:
            ensure_dir(PROJECT_ROOT / "results")
            pd.DataFrame(detections).to_csv(PROJECT_ROOT / "results" / "xai4heat_detected_columns.csv", index=False)
            return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "substation_id"])
    return _load_generic_dataset("xai4heat", raw_dir)


def load_aalborg(raw_dir: str | Path) -> pd.DataFrame:
    df = _load_generic_dataset("aalborg", raw_dir)
    if "plant_id" in df.columns:
        df = (
            df.groupby("timestamp", as_index=False)
            .agg(
                heat_load_kw=("heat_load_kw", "sum"),
                supply_temp_C=("supply_temp_C", "mean"),
                return_temp_C=("return_temp_C", "mean"),
                ambient_temp_C=("ambient_temp_C", "mean"),
            )
        )
        df["source_dataset"] = "aalborg"
    return df


def load_dataset_by_name(dataset_name: str) -> pd.DataFrame:
    registry = DataRegistry()
    info = registry.get(dataset_name)
    loaders = {
        "sonderborg": load_sonderborg,
        "flensburg": load_flensburg,
        "xai4heat": load_xai4heat,
        "aalborg": load_aalborg,
    }
    loader = loaders.get(info.loader)
    if loader is None:
        raise KeyError(f"No loader registered for {dataset_name} ({info.loader})")
    return loader(info.raw_dir)
