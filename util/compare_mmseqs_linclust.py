#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

LINCLUST_MIN_SEQ_ID = "0.9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone two MMseqs2 repos at two commits, build both, run linclust with "
            f"--min-seq-id {LINCLUST_MIN_SEQ_ID} on the same FASTA file, and compare "
            "the resulting cluster memberships. repo1 is built with the default "
            "configuration, repo2 is built with -DMMSEQS_INT64_IDS=1."
        )
    )
    parser.add_argument("--repo1", type=str, help="First git repository URL.")
    parser.add_argument(
        "--commit1",
        type=str,
        help="Commit or ref to check out in the first repository.",
    )
    parser.add_argument("--repo2", type=str, help="Second git repository URL.")
    parser.add_argument(
        "--commit2",
        type=str,
        help="Commit or ref to check out in the second repository.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Directory for clones, builds, outputs, and reports.",
    )
    parser.add_argument(
        "--input_fasta",
        type=Path,
        required=True,
        help="Single FASTA file used for all runs.",
    )
    parser.add_argument("--nthreads", type=int, help="Threads passed to MMseqs2.")
    parser.add_argument(
        "--njobs", type=int, help="Parallel jobs passed to cmake --build."
    )
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable_cwd = cwd if cwd is not None else Path.cwd()
    print(f"\n[{printable_cwd}]$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_repo(repo_url: str, commit: str, clone_dir: Path) -> None:
    # Start from a clean clone so the script is reproducible and self-contained.
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    clone_dir.parent.mkdir(parents=True, exist_ok=True)

    run(["git", "clone", repo_url, str(clone_dir)])
    run(["git", "checkout", "--detach", commit], cwd=clone_dir)
    run(["git", "reset", "--hard", commit], cwd=clone_dir)


def find_mmseqs_binary(build_dir: Path) -> Path:
    for candidate in (
        build_dir / "src" / "mmseqs",
        build_dir / "bin" / "mmseqs",
        build_dir / "mmseqs",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find built mmseqs binary under {build_dir}")


def build_repo(
    source_dir: Path, build_dir: Path, njobs: int, *, enable_int64_ids: bool
) -> Path:
    reset_dir(build_dir)
    configure_command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if enable_int64_ids:
        configure_command.append("-DMMSEQS_INT64_IDS=1")
    run(configure_command)
    run(["cmake", "--build", str(build_dir), "--target", "mmseqs", "-j", str(njobs)])
    return find_mmseqs_binary(build_dir)


def run_linclust(
    mmseqs_binary: Path,
    input_fasta: Path,
    output_dir: Path,
    nthreads: int,
) -> Path:
    reset_dir(output_dir)
    input_db = output_dir / "input"
    cluster_db = output_dir / "cluster"
    tmp_dir = output_dir / "tmp"
    cluster_tsv = output_dir / "cluster.tsv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            str(mmseqs_binary),
            "createdb",
            str(input_fasta),
            str(input_db),
        ]
    )

    run(
        [
            str(mmseqs_binary),
            "linclust",
            str(input_db),
            str(cluster_db),
            str(tmp_dir),
            "--min-seq-id",
            LINCLUST_MIN_SEQ_ID,
            "--threads",
            str(nthreads),
        ]
    )

    run(
        [
            str(mmseqs_binary),
            "createtsv",
            str(input_db),
            str(input_db),
            str(cluster_db),
            str(cluster_tsv),
            "--threads",
            str(nthreads),
        ]
    )

    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"Expected cluster output {cluster_tsv} was not created"
        )
    return cluster_tsv


def normalize_clusters(
    cluster_tsv: Path,
) -> tuple[list[tuple[str, ...]], dict[str, int | str]]:
    # Raw MMseqs2 TSV order can change between runs, so compare sorted member sets instead.
    clusters_by_representative: dict[str, set[str]] = defaultdict(set)
    sequence_ids: set[str] = set()

    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise RuntimeError(
                    f"Malformed cluster line {line_number} in {cluster_tsv}: {raw_line!r}"
                )

            representative, member = fields[0], fields[1]
            clusters_by_representative[representative].add(representative)
            clusters_by_representative[representative].add(member)
            sequence_ids.add(representative)
            sequence_ids.add(member)

    normalized_clusters = sorted(
        tuple(sorted(members)) for members in clusters_by_representative.values()
    )
    digest = hashlib.sha256()
    for members in normalized_clusters:
        digest.update("\t".join(members).encode("utf-8"))
        digest.update(b"\n")

    summary: dict[str, int | str] = {
        "cluster_count": len(normalized_clusters),
        "sequence_count": len(sequence_ids),
        "normalized_sha256": digest.hexdigest(),
        "raw_cluster_tsv_sha256": sha256_file(cluster_tsv),
    }
    return normalized_clusters, summary


def write_report(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def summarize_linclust(
    repo1_name: str,
    repo2_name: str,
    repo1_binary: Path,
    repo2_binary: Path,
    input_fasta: Path,
    workdir: Path,
    nthreads: int,
) -> dict:
    repo1_cluster_tsv = run_linclust(
        repo1_binary,
        input_fasta,
        workdir / "results" / repo1_name / "linclust",
        nthreads,
    )
    repo2_cluster_tsv = run_linclust(
        repo2_binary,
        input_fasta,
        workdir / "results" / repo2_name / "linclust",
        nthreads,
    )

    repo1_clusters, repo1_summary = normalize_clusters(repo1_cluster_tsv)
    repo2_clusters, repo2_summary = normalize_clusters(repo2_cluster_tsv)

    return {
        "equivalent_cluster_membership": repo1_clusters == repo2_clusters,
        repo1_name: {
            "binary": str(repo1_binary),
            "cluster_tsv": str(repo1_cluster_tsv),
            **repo1_summary,
        },
        repo2_name: {
            "binary": str(repo2_binary),
            "cluster_tsv": str(repo2_cluster_tsv),
            **repo2_summary,
        },
    }


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    input_fasta = args.input_fasta.resolve()

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    repo1_name = "repo1"
    repo2_name = "repo2"

    repo1_clone_dir = workdir / "clones" / repo1_name
    repo2_clone_dir = workdir / "clones" / repo2_name
    repo1_build_dir = workdir / "builds" / repo1_name
    repo2_build_dir = workdir / "builds" / repo2_name

    clone_repo(args.repo1, args.commit1, repo1_clone_dir)
    clone_repo(args.repo2, args.commit2, repo2_clone_dir)

    # repo1 is the default-width reference build, repo2 is the int64 build under test.
    repo1_binary = build_repo(
        repo1_clone_dir, repo1_build_dir, args.njobs, enable_int64_ids=False
    )
    repo2_binary = build_repo(
        repo2_clone_dir, repo2_build_dir, args.njobs, enable_int64_ids=True
    )

    linclust_report = summarize_linclust(
        repo1_name,
        repo2_name,
        repo1_binary,
        repo2_binary,
        input_fasta,
        workdir,
        args.nthreads,
    )

    report = {
        "repo1": {"url": args.repo1, "commit": args.commit1},
        "repo2": {"url": args.repo2, "commit": args.commit2},
        "workdir": str(workdir),
        "input_fasta": str(input_fasta),
        "nthreads": args.nthreads,
        "njobs": args.njobs,
        "workflow": "linclust",
        "min_seq_id": float(LINCLUST_MIN_SEQ_ID),
        "build_modes": {
            "repo1": "default",
            "repo2": "MMSEQS_INT64_IDS=1",
        },
        "linclust": linclust_report,
    }

    report_path = workdir / "reports" / "comparison.json"
    write_report(report_path, report)

    all_equal = linclust_report["equivalent_cluster_membership"]

    print("\nComparison report written to:", report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
