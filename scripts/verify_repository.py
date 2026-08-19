#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_TITLE = "Real-Data-Assisted Thermal State Estimation in District Heating Networks: An Evidence-Separated Benchmark with Simulator-Assisted Hydraulics"
EXPECTED_WEATHER_SHA256 = "36ca9f9d93b92136649536f94885cf4bf049a0dd4fea88ad494030627283f7c5"
EXPECTED_CANONICAL_SHA256 = "35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e"
EXPECTED_PT = 56
EXPECTED_NPZ = 3
EXPECTED_MAIN_FIGURES = 10
REQUIRED_RESULTS = [
    "baseline_comparison_final.csv",
    "physics_consistency_comparison_final.csv",
    "calibration_metrics.csv",
    "proxy_causality_audit.csv",
    "repeated_seed_statistics.csv",
    "multi_window_three_seed_summary.csv",
    "second_chronological_window_summary.csv",
    "flensburg_measured_only_validation.csv",
    "moving_block_bootstrap_ci.csv",
    "parameter_identifiability_sensitivity.csv",
    "heat_loss_profile_metrics.csv",
    "energy_balance_time_series.csv",
    "thermo_hydraulic_robustness.csv",
    "uncertainty_quantification_metrics.csv",
    "anomaly_detection_metrics_improved.csv",
    "anomaly_detection_timeseries_improved.csv",
    "operational_energy_impact_timeseries.csv",
    "concept_model_value_rank_matrix.csv",
    "sensor_layout_definitions_table.csv",
    "data_availability_report.csv",
]
TEXT_EXT = {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".py", ".cff", ".toml"}
ABS_PATH = re.compile(r"(?i)(?:[A-Z]:\\Users\\|/Users/|/home/)[^\r\n,;\"]+")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cond: bool, label: str, details: str = "") -> tuple[bool, str]:
    return cond, label + (f" — {details}" if details else "")


def verify_manifest() -> tuple[bool, str]:
    path = ROOT / "release_manifest.sha256"
    if not path.exists():
        return check(False, "release manifest", "missing release_manifest.sha256")
    bad=[]; count=0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try: expected, rel = line.split("  ", 1)
        except ValueError: return check(False,"release manifest",f"malformed line: {line[:80]}")
        p=ROOT/rel; count+=1
        if not p.exists() or digest(p) != expected: bad.append(rel)
    return check(not bad, "release manifest", f"{count} files checked" if not bad else f"mismatch/missing: {bad[:5]}")


def verify_csvs() -> tuple[bool, str]:
    bad=[]; count=0
    for p in (ROOT/"results").glob("*.csv"):
        try:
            with p.open(newline="", encoding="utf-8-sig") as f:
                reader=csv.reader(f); header=next(reader, None)
                if not header: raise ValueError("empty header")
                for _ in reader: pass
            count+=1
        except Exception as e: bad.append(f"{p.name}: {e}")
    return check(not bad, "CSV parse", f"{count} result CSVs" if not bad else "; ".join(bad[:3]))


def verify_json() -> tuple[bool, str]:
    bad=[]; count=0
    for p in ROOT.rglob("*.json"):
        try: json.loads(p.read_text(encoding="utf-8")); count+=1
        except Exception as e: bad.append(f"{p.relative_to(ROOT)}: {e}")
    return check(not bad, "JSON parse", f"{count} JSON files" if not bad else "; ".join(bad[:3]))


def verify_paths() -> tuple[bool, str]:
    bad=[]
    for p in ROOT.rglob("*"):
        if p.resolve() == Path(__file__).resolve(): continue
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            txt=p.read_text(encoding="utf-8", errors="ignore")
            if ABS_PATH.search(txt): bad.append(str(p.relative_to(ROOT)))
    return check(not bad, "path sanitization", "no user-home absolute paths" if not bad else f"found in {bad[:5]}")


def verify_identity() -> tuple[bool,str]:
    files=[ROOT/'README.md',ROOT/'PAPER_SCOPE.md',ROOT/'CITATION.cff',ROOT/'RELEASE.json']
    missing=[str(p.relative_to(ROOT)) for p in files if not p.exists()]
    if missing: return check(False,'paper identity',f'missing {missing}')
    bad=[str(p.relative_to(ROOT)) for p in files if PAPER_TITLE not in p.read_text(encoding='utf-8',errors='ignore')]
    return check(not bad,'paper identity',PAPER_TITLE if not bad else f'title absent from {bad}')


def verify_measured_data_exclusion() -> tuple[bool,str]:
    bad=[]
    for rel in ['data/raw','data/processed','data/locked']:
        d=ROOT/rel
        if d.exists():
            for p in d.rglob('*'):
                if p.is_file() and p.name!='.gitkeep': bad.append(str(p.relative_to(ROOT)))
    return check(not bad,'measured-data redistribution policy','third-party measured rows absent' if not bad else f'unexpected files: {bad[:5]}')


def verify_figures() -> tuple[bool,str]:
    d=ROOT/'figures'/'main'
    stems=set()
    if d.exists():
        for p in d.glob('fig*.svg'):
            m=re.match(r'(fig\d\d)_',p.name)
            if m: stems.add(m.group(1))
    formats_ok=all(any(d.glob(f'{fig}_*.{ext}')) for fig in sorted(stems) for ext in ['svg','pdf','png']) if stems else False
    prov=(d/'figure_provenance.csv').exists()
    ok=len(stems)==EXPECTED_MAIN_FIGURES and formats_ok and prov
    return check(ok,'main figure set',f'{len(stems)} figures; SVG/PDF/PNG + provenance')


def verify_protocol_metadata() -> tuple[bool,str]:
    p=ROOT/'results'/'gap_handling_audit.json'
    try: d=json.loads(p.read_text())
    except Exception as e: return check(False,'chronological protocol',repr(e))
    expected=(d.get('retained_timestamp_count')==768 and d.get('gap_start_index')==62 and d.get('eligible_contiguous_window_count')==746 and d.get('training_window_starts')==522 and d.get('validation_window_starts')==111 and d.get('test_window_starts')==91)
    return check(expected,'chronological protocol','768 samples; restart 62; windows 746 -> 522/111/91')


def main() -> int:
    ap=argparse.ArgumentParser(description="Verify exact-paper public release integrity and evidence boundaries.")
    ap.add_argument("--full", action="store_true", help="also load checkpoints/NPZ files and import source modules (requires dependencies)")
    args=ap.parse_args()
    tests=[]
    for name in ["README.md","LICENSE","CITATION.cff","DATA_LICENSES.md","REPRODUCIBILITY.md","CHANGELOG.md","PAPER_SCOPE.md","CLAIM_EVIDENCE_MAP.md","RELEASE.json"]:
        tests.append(check((ROOT/name).exists(), f"metadata {name}"))
    tests.append(verify_identity())
    for name in REQUIRED_RESULTS:
        tests.append(check((ROOT/"results"/name).exists(), f"result {name}"))
    weather=ROOT/"data"/"external_weather"/"sonderborg_era5_land_2016_2019_hourly.csv"
    tests.append(check(weather.exists() and digest(weather)==EXPECTED_WEATHER_SHA256, "ERA5-Land hash", EXPECTED_WEATHER_SHA256))
    tests.append(verify_measured_data_exclusion())
    pts=list((ROOT/"results").glob("*.pt")); npzs=list((ROOT/"results").glob("*.npz"))
    tests.append(check(len(pts)==EXPECTED_PT, "checkpoint count", str(len(pts))))
    tests.append(check(len(npzs)==EXPECTED_NPZ, "NPZ count", str(len(npzs))))
    tests += [verify_figures(), verify_protocol_metadata(), verify_csvs(), verify_json(), verify_paths(), verify_manifest()]
    if args.full:
        try:
            import numpy as np
            for p in npzs:
                with np.load(p, allow_pickle=False) as z: _=list(z.files)
            tests.append(check(True,"NPZ load",f"{len(npzs)} files"))
        except Exception as e: tests.append(check(False,"NPZ load",repr(e)))
        try:
            import torch
            for p in pts: torch.load(p, map_location="cpu", weights_only=True)
            tests.append(check(True,"checkpoint load",f"{len(pts)} state_dict files"))
        except Exception as e: tests.append(check(False,"checkpoint load",repr(e)))
        try:
            import importlib, pkgutil
            sys.path.insert(0,str(ROOT)); failed=[]; n=0
            for mod in pkgutil.iter_modules([str(ROOT/"src")]):
                n+=1
                try: importlib.import_module("src."+mod.name)
                except Exception as e: failed.append(f"{mod.name}: {type(e).__name__}: {e}")
            tests.append(check(not failed,"source import",f"{n} modules" if not failed else "; ".join(failed[:4])))
        except Exception as e: tests.append(check(False,"source import",repr(e)))
    ok=True
    for passed,label in tests:
        print(("[PASS] " if passed else "[FAIL] ")+label); ok &= passed
    print("\n" + ("REPOSITORY VERIFICATION: PASS" if ok else "REPOSITORY VERIFICATION: FAIL"))
    return 0 if ok else 1
if __name__ == "__main__": raise SystemExit(main())
