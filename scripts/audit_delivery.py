"""Read-only, standard-library asset inventory. Never trains or starts a simulator."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect_asset(root, spec, tracked):
    files, missing = [], []
    for name in spec["paths"]:
        target = root / name
        candidates = sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else [target]
        if not target.exists():
            missing.append(name)
            continue
        for path in candidates:
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                continue
            relative = path.relative_to(root).as_posix()
            files.append({"path": relative, "bytes": path.stat().st_size,
                          "sha256": sha256(path), "in_git": relative in tracked})
    expected = spec.get("sha256")
    if "expected" in spec:
        ref = spec["expected"]
        expected = json.loads((root / ref["path"]).read_text(encoding="utf-8"))
        for key in ref["keys"]:
            expected = expected[key]
    matches = None if expected is None or not files else len(files) == 1 and files[0]["sha256"] == expected
    bundle = "".join(f"{f['path']}\t{f['sha256']}\n" for f in files).encode()
    return {**spec, "present": not missing and bool(files), "missing": missing,
            "file_count": len(files), "bytes": sum(f["bytes"] for f in files),
            "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "expected_sha256_match": matches, "files": files}


def audit(root):
    tracked = set(subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines())
    specs = json.loads((root / "configs/delivery_assets.json").read_text(encoding="utf-8"))["assets"]
    assets = [inspect_asset(root, spec, tracked) for spec in specs]
    dependencies = []
    for requirement in tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]:
        name, expected = requirement.split("==")
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        dependencies.append({"package": name, "expected": expected, "installed": actual,
                             "matches_direct_pin": actual is not None and actual.split("+")[0] == expected})
    compose = (root / "infra/boptest.compose.yml").read_text(encoding="utf-8")
    return {"audit_schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "scope": "current local working files; not a clean-environment or redistribution certificate",
            "hash_method": "raw file SHA256; bundle SHA256 of ordered path-tab-filehash-newline entries",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "executable": str(Path(sys.executable).relative_to(root)) if Path(sys.executable).is_relative_to(root) else "external interpreter",
                            "dependencies": dependencies},
            "compose_images": sorted(set(re.findall(r"^\s+image:\s*(\S+)", compose, re.M))),
            "build_base_images": re.findall(r"^FROM (\S+)", (root / "infra/storage.Dockerfile").read_text(), re.M),
            "assets": assets,
            "summary": {"assets": len(assets), "missing_assets": [a["id"] for a in assets if not a["present"]],
                        "hash_mismatches": [a["id"] for a in assets if a["expected_sha256_match"] is False],
                        "direct_dependency_mismatches": [d["package"] for d in dependencies if not d["matches_direct_pin"]]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/delivery-audit/assets.json")
    args = parser.parse_args()
    result = audit(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    if result["summary"]["hash_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
