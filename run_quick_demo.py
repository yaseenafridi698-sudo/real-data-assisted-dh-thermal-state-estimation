#!/usr/bin/env python3
"""Software smoke/demo workflow. Its fallback-synthetic outputs are not paper evidence."""
from __future__ import annotations
import os
from src.config import ensure_project_dirs, load_config
from src.study_workflow import run_demo_workflow

def main() -> None:
    ensure_project_dirs(); os.environ['CAUSAL_SKIP_ASSET_GENERATION']='1'
    result=run_demo_workflow(load_config())
    print(f"Quick demo completed. fallback_synthetic={result['used_fallback']}")

if __name__=='__main__': main()
