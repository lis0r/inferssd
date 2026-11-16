#!/usr/bin/env python3
"""Bounded source/schema monitoring; offline replay by default, never loads weights."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "framework/tests/fixtures/gsq-rco/iq2-xxs-allocation.txt"
URL = (
    "https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/"
    "resolve/main/tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.rco-allocation.txt"
)
EXPECTED_SHA = "305e14351d8f116f411e0370a9a6f73f0cde2941b7a4650b0fb35be00356a91b"
EXPECTED_COUNTS = {
    "F32": 531, "BF16": 67, "Q8_0": 80, "Q5_K": 244,
    "IQ4_NL": 1, "IQ2_XS": 18, "IQ2_XXS": 42, "IQ1_M": 60,
}


def allocation(raw: bytes) -> dict[str, str]:
    tensors: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, kind = line.rsplit(": ", 1)
        if name in tensors or kind not in EXPECTED_COUNTS:
            raise ValueError("duplicate tensor or unknown storage type: " + name)
        tensors[name] = kind
    if dict(collections.Counter(tensors.values())) != EXPECTED_COUNTS:
        raise ValueError("per-type tensor coverage differs from the 1,043-tensor baseline")
    if tensors.get("per_layer_token_embd.weight") != "IQ4_NL":
        raise ValueError("missing shared IQ4_NL Engram table")
    for layer in range(40):
        kinds = [tensors[f"blk.{layer}.ffn_{role}_exps.weight"] for role in ("up", "gate", "down")]
        if kinds[0] != kinds[1]:
            raise ValueError("up/gate types disagree in layer " + str(layer))
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="make one bounded request for the live allocation")
    parser.add_argument("--output", type=Path, required=True, help="parent of dated, immutable result directories")
    args = parser.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    directory = args.output / now.strftime("%Y%m%dT%H%M%S.%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    result: dict[str, object] = {
        "schema": "inferssd-gsq-rco-source/1", "observed_at": now.isoformat(),
        "source": URL, "baseline_revision": "65d53bc8be65123e255ff02f3ded958b6dbf54fd",
        "baseline_sha256": EXPECTED_SHA, "source_availability": "not_requested",
        "source_publication_time": None, "status": "pending",
        "scope": "allocation availability/schema/identity; no payload, CUDA or inference certification",
    }
    started = time.monotonic()
    try:
        result["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5
        ).strip()
        baseline = BASELINE.read_bytes()
        if hashlib.sha256(baseline).hexdigest() != EXPECTED_SHA:
            raise ValueError("cached evidence hash mismatch")
        expected = allocation(baseline)
        raw = baseline
        if args.refresh:
            result["source_availability"] = "pending"
            try:
                with urllib.request.urlopen(URL, timeout=20) as response:
                    raw = response.read(256 * 1024 + 1)
                    result["source_last_modified"] = response.headers.get("Last-Modified")
                result["source_availability"] = "available"
            except (OSError, TimeoutError):
                result["source_availability"] = "unavailable"
                raise
            if len(raw) > 256 * 1024:
                raise ValueError("allocation response exceeds the 256 KiB bound")
        (directory / "allocation.txt").write_bytes(raw)
        result["observed_sha256"] = hashlib.sha256(raw).hexdigest()
        result["response_bytes"] = len(raw)
        actual = allocation(raw)
        if actual != expected:
            raise ValueError("tensor allocation changed; a reviewed baseline update is required")
        result["tensor_count"] = len(actual)
        result["type_counts"] = dict(collections.Counter(actual.values()))
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "unavailable" if result["source_availability"] == "unavailable" else "fail"
        result["diagnostic"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.monotonic() - started
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
