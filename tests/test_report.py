from pathlib import Path


def test_reports_directory_exists():
    reports_dir = Path("reports")

    assert reports_dir.exists()