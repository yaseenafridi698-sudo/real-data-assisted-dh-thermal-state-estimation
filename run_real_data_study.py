#!/usr/bin/env python3
"""Run the full scientific training/evaluation workflow for the paper.

The public repository does not redistribute third-party measured datasets.
Install the checksum-pinned canonical Sonderborg processed artifact first; add
external datasets locally when external-validation regeneration is required.
"""
from __future__ import annotations
import argparse, os
from src.config import PROJECT_ROOT, ensure_project_dirs, load_config
from src.study_workflow import run_real_data_workflow


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--generate-assets', action='store_true', help='also run legacy paper-asset generation after the scientific workflow')
    args=ap.parse_args()
    ensure_project_dirs(); config=load_config()
    rel=config.get('real_data',{}).get('canonical_sonderborg_processed_path','data/locked/sonderborg_processed_18703.csv')
    canonical=PROJECT_ROOT/str(rel)
    raw_dir=PROJECT_ROOT/'data'/'raw'/'sonderborg'
    if not canonical.exists() and not any(p.is_file() and p.name!='.gitkeep' for p in raw_dir.glob('*')):
        raise SystemExit('Canonical Sonderborg input is not installed. Run: python scripts/install_canonical_data.py /path/to/sonderborg_processed_18703.csv')
    if not args.generate_assets:
        os.environ['CAUSAL_SKIP_ASSET_GENERATION']='1'
    result=run_real_data_workflow(config)
    if not result.get('ran_main_results',False):
        raise SystemExit('Main real-data workflow did not run; check locally installed inputs.')
    print('Real-data scientific workflow completed.')

if __name__=='__main__': main()
