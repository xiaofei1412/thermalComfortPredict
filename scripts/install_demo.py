"""Install the checksum-verified real simulation sample without overwriting history."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sample(root=ROOT):
    folder = root / "examples/workbench"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    compressed = (folder / "0a2e84c3cb66.json.gz").read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["gzip_sha256"]:
        raise ValueError("Bundled sample archive checksum mismatch")
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != manifest["original_session_sha256"]:
        raise ValueError("Bundled original session checksum mismatch")
    session = json.loads(raw)
    if session["id"] != "0a2e84c3cb66" or session["status"] != "completed" or session["schema"] != 2:
        raise ValueError("Unsupported demonstration session")
    return session, raw, manifest


def install(root=ROOT, store=None):
    session, raw, _ = sample(root)
    store = store or root / "outputs/workbench/sessions"
    destination = store / (session["id"] + ".json")
    store.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError(f"Existing history differs; preserved without overwrite: {destination}")
    else:
        destination.write_bytes(raw)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path)
    args = parser.parse_args()
    print(f"Real simulation replay sample ready: {install(store=args.store)}")


if __name__ == "__main__":
    main()
