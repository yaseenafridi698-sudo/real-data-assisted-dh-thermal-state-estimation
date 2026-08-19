#!/usr/bin/env python3
"""Install an author-held canonical Sønderborg processed file after hash verification.

This does not download or redistribute the measured dataset. It only verifies a
local file supplied by the user and copies it into the path expected by the
historical study configuration.
"""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED="35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e"
EXPECTED_ROWS=18703

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('file', type=Path); args=ap.parse_args()
    if not args.file.is_file(): raise SystemExit(f'File not found: {args.file}')
    got=sha256(args.file)
    if got != EXPECTED: raise SystemExit(f'SHA-256 mismatch. Expected {EXPECTED}, got {got}')
    # count lines without pandas; CSV has one header row
    with args.file.open('rb') as f: rows=sum(1 for _ in f)-1
    if rows != EXPECTED_ROWS: raise SystemExit(f'Row-count mismatch. Expected {EXPECTED_ROWS}, got {rows}')
    dst=ROOT/'data'/'locked'/'sonderborg_processed_18703.csv'; dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.file,dst)
    print(f'PASS: installed canonical file at {dst.relative_to(ROOT)}')
if __name__=='__main__': main()
