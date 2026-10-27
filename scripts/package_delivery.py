"""Build and read-back verify source delivery or a private local evidence archive."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024*1024), b""):
        h.update(chunk)
    return h.hexdigest()


def public_paths(root):
    paths = [p for p in root.glob("*.md") if p.is_file()]
    paths += [root/n for n in ("pyproject.toml", ".gitattributes", ".gitignore", ".dockerignore") if (root/n).exists()]
    for name in ("thermal_comfort", "workbench", "scripts", "configs", "requirements", "examples", "docs", "reports", "tests", "infra", ".github"):
        paths += [p for p in (root/name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths += [root/n for n in ("data/README.md", "data/manifest.json", "legacy/README.md") if (root/n).exists()]
    return sorted(set(paths))


def private_paths(root):
    prefixes = ("baseline", "excitation", "dynamics", "greybox", "action-response", "mpc", "recovery-", "solar-", "generalization-", "office-", "terminal-", "native-", "simulation", "workbench", "residential-c")
    paths = []
    for folder in (root/"outputs").iterdir():
        if folder.is_dir() and folder.name.startswith(prefixes):
            paths += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".env") and p.suffix != ".pyc"]
    for name in ("data/raw/ashrae_db2.csv", "data/processed/cleaned_asds4.csv", "data/manifest.json"):
        if (root/name).exists(): paths.append(root/name)
    return sorted(set(paths))


def package(root, kind, destination):
    paths = public_paths(root) if kind == "source" else private_paths(root)
    if not paths:
        raise ValueError("No delivery files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")) if kind == "source" else set()
    rows = []
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            name = path.relative_to(root).as_posix()
            if kind == "source":
                if name not in tracked:
                    raise ValueError(f"Stage and review source before packaging: {name}")
                data = subprocess.check_output(["git", "show", ":"+name], cwd=root)
                expected, size = hashlib.sha256(data).hexdigest(), len(data)
                archive.writestr(name, data)
            else:
                with path.open("rb") as stream: expected = sha_stream(stream)
                size = path.stat().st_size
                archive.write(path, name)
            rows.append({"path": name, "bytes": size, "sha256": expected})
    with zipfile.ZipFile(destination) as archive:
        for row in rows:
            with archive.open(row["path"]) as stream:
                if sha_stream(stream) != row["sha256"]:
                    raise ValueError("Archive read-back mismatch")
    with destination.open("rb") as stream: archive_hash = sha_stream(stream)
    result = {"kind": kind, "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "archive": destination.name, "archive_sha256": archive_hash, "archive_bytes": destination.stat().st_size,
              "file_count": len(rows), "uncompressed_bytes": sum(r["bytes"] for r in rows),
              "read_back_verified": True, "files": rows,
              "scope": "source workspace, no historical CSV/weights/FMU" if kind == "source" else "private local copy only; not off-site backup or a redistribution license"}
    destination.with_suffix(".manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["source", "private"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(ROOT, args.kind, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False), flush=True)
