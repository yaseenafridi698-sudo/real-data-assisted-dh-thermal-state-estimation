from __future__ import annotations

from src.config import ensure_project_dirs, PROJECT_ROOT
from src.data_download import download_all_datasets


def main() -> None:
    ensure_project_dirs()
    statuses = download_all_datasets()
    print("\nDownload summary")
    for name, ok in statuses.items():
        print(f"  {name}: {'available/downloaded' if ok else 'manual action needed'}")
    print(f"Data availability report: {PROJECT_ROOT / 'results' / 'data_availability_report.csv'}")

if __name__ == "__main__":
    main()
