from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool = True) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
    }
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-raw", action="store_true")
    args = parser.parse_args()

    required = [
        ROOT / "data/interim/cold_landmarks_all.parquet",
        ROOT / "data/interim/cold_landmarks_all_strict.parquet",
        ROOT / "artifacts/revision_round1/operational_policy_audit.csv",
        ROOT / "artifacts/revision_round1/candidate_sequence_audit.csv",
        ROOT / "artifacts/revision_round1/predictive_metrics.csv",
        ROOT / "artifacts/revision_round1/reliability_bins.csv",
        ROOT / "artifacts/revision_round1/coverage/coverage_summary.csv",
        ROOT / "artifacts/revision_round1/censoring/episode_end_reasons.csv",
        ROOT / "artifacts/revision_round1/censoring/risk_set_censoring_audit.csv",
        ROOT / "artifacts/revision_round1/censoring/disappearance_sensitivity.csv",
        ROOT / "artifacts/revision_round1/claim_verification_round1.json",
        ROOT / "artifacts/trace_simulation/design.json",
        ROOT / "artifacts/trace_simulation/claim_verification_trace_simulation.json",
        ROOT / "artifacts/trace_simulation/replication_results.csv",
        ROOT / "artifacts/trace_simulation/scenario_summary.csv",
        ROOT / "artifacts/trace_simulation/selected_actions.parquet",
        ROOT / "artifacts/placement_analysis/design.json",
        ROOT / "artifacts/placement_analysis/reconstruction_audit.json",
        ROOT / "artifacts/placement_analysis/claim_verification_placement.json",
        ROOT / "artifacts/placement_analysis/scenario_summary.csv",
        ROOT / "artifacts/placement_analysis/daily_results.csv",
        ROOT / "artifacts/placement_analysis/action_blocks.parquet",
        ROOT / "figures/figure_round1_revision.pdf",
        ROOT / "figures/figure_round1_revision.svg",
        ROOT / "figures/figure_trace_simulation.pdf",
        ROOT / "figures/figure_trace_simulation.svg",
        ROOT / "figures/figure_trace_simulation_source.csv",
        ROOT / "figures/figure_placement_analysis.pdf",
        ROOT / "figures/figure_placement_analysis.svg",
        ROOT / "figures/figure_placement_analysis_source.csv",
        ROOT / "paper/figures/coldguard_method.drawio",
        ROOT / "paper/figures/coldguard_method.pdf",
        ROOT / "paper/manuscript_submission.md",
        ROOT / "paper/jnsm_latex/main.tex",
        ROOT / "paper/jnsm_latex/body.tex",
        ROOT / "paper/jnsm_latex/supplementary.tex",
        ROOT / "paper/jnsm_latex/references.bib",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required reproducibility outputs: {missing}")

    source_files = sorted([
        ROOT / "pyproject.toml", ROOT / "uv.lock",
        *ROOT.glob("*.md"),
        *ROOT.glob("literature/*.md"), *ROOT.glob("literature/*.bib"),
        *ROOT.glob("config/*.yaml"), *ROOT.glob("src/**/*.py"),
        *ROOT.glob("tests/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("scripts/*.sh"),
        *ROOT.glob("paper/*.md"), *ROOT.glob("paper/jnsm_latex/*.tex"),
        *ROOT.glob("paper/figures/*.drawio"),
    ])
    source_files = [path for path in source_files if path.exists()]
    destination = ROOT / "artifacts/reproducibility_manifest.json"
    result_files = sorted([
        *ROOT.glob("artifacts/**/*.csv"), *ROOT.glob("artifacts/**/*.json"),
        *ROOT.glob("artifacts/**/report.md"), *ROOT.glob("figures/*_source.csv"),
        *ROOT.glob("figures/*_caption.md"), *ROOT.glob("figures/*_manifest.json"),
        *ROOT.glob("literature/*.csv"),
    ])
    # A manifest cannot contain a stable hash of itself. Excluding the destination
    # keeps every recorded digest independently verifiable after the file is written.
    result_files = [path for path in result_files if path != destination]
    raw_archives = sorted(ROOT.glob("data/raw/*.zip"))
    packages = {}
    for name in ["coldguard", "duckdb", "joblib", "matplotlib", "numpy", "pandas", "pyarrow", "PyYAML", "scikit-learn", "scipy", "simpy"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    manifest = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "raw_archives": [file_record(path, include_hash=args.hash_raw) for path in raw_archives],
        "source_files": [file_record(path) for path in source_files],
        "result_files": [file_record(path) for path in result_files],
        "required_outputs": [str(path.relative_to(ROOT)) for path in required],
        "raw_content_hashing": bool(args.hash_raw),
    }
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
