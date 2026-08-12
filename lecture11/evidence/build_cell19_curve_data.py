#!/usr/bin/env python3
"""Rebuild or verify the native L11 train/held-out NLL curve data.

The plotted CSV is a bit-exact decimal round trip of ``train_history`` and
``valid_history`` from the executed course notebook. Cell 19 only draws those
histories; cells 4, 6, and 18 define the corpus, windows, and seed-11 training
run that produces them. Pinning the notebook hash makes a changed experiment
an explicit review event rather than silently changing the lecture evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/L11/01_char_mlp_language_model.ipynb"
OUTPUT = Path(__file__).with_name("cell19_train_heldout_nll.csv")
NOTEBOOK_SHA256 = "0bafad82d1055c4d23622491746a832228b234f35407e4cf2c143fc318b875f5"
EXECUTED_CELLS = (4, 6, 18)


def expected_csv() -> str:
    raw = NOTEBOOK.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != NOTEBOOK_SHA256:
        raise RuntimeError(
            "Notebook changed; review the experiment and update the pinned hash "
            f"before rebuilding evidence (expected {NOTEBOOK_SHA256}, got {digest})."
        )

    notebook = json.loads(raw)
    namespace = {"np": np, "math": math, "SEED": 11}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_index in EXECUTED_CELLS:
            source = "".join(notebook["cells"][cell_index]["source"])
            exec(compile(source, f"{NOTEBOOK.name}:cell-{cell_index}", "exec"), namespace)

    train = namespace["train_history"]
    heldout = namespace["valid_history"]
    if len(train) != 300 or len(heldout) != 300:
        raise RuntimeError("Expected exactly 300 post-update NLL values per split.")

    lines = ["update,train_nll,heldout_nll"]
    lines.extend(
        f"{update},{float(train_nll)!r},{float(heldout_nll)!r}"
        for update, (train_nll, heldout_nll) in enumerate(zip(train, heldout), 1)
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the checked CSV; without this flag, verify it byte-for-byte",
    )
    args = parser.parse_args()

    expected = expected_csv()
    if args.write:
        OUTPUT.write_text(expected, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(ROOT)}")
        return

    actual = OUTPUT.read_text(encoding="utf-8")
    if actual != expected:
        raise SystemExit(
            f"FAIL: {OUTPUT.relative_to(ROOT)} is stale; rerun with --write after review"
        )
    print(
        "PASS: CSV matches notebook cells 4/6/18 exactly "
        "(seed 11, 300 full-batch Adam updates, 56/30 targets)"
    )


if __name__ == "__main__":
    main()
