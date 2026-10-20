"""Build both caches and print the Step 0 count check.

    python -m src.build_cache --fafb-edges data/raw/<buhmann per-neuropil table> \
        --malecns-edges data/raw/<bulk edge file> [--malecns-neurons data/raw/<neuron file>] \
        [--only fafb|malecns] [--force] [--super-class-map sensory_ascending=ascending]

--super-class-map is for values the group has not agreed on yet; it is PROVISIONAL and prints a warning.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.fetch_fafb import ANNOTATIONS_URL, fetch_fafb  # noqa: E402
from src.fetch_malecns import fetch_malecns  # noqa: E402
from src.validators import assert_interchangeable  # noqa: E402

EXPECTED = {"fafb neurons": 139_248, "fafb edges >= 5": 2_700_513, "malecns neurons": 165_122}


def step0(name: str, neurons: pd.DataFrame, edges: pd.DataFrame, self_conn: int | None) -> None:
    n5 = int((edges["syn_count"] >= 5).sum())
    rows = [(f"{name} neurons", len(neurons)), (f"{name} edges >= 5", n5)]
    print(f"\nStep 0 — {name}")
    for label, got in rows:
        exp = EXPECTED.get(label)
        flag = "" if exp is None else ("  OK" if got == exp else f"  DIFFERENT (expected {exp:,})")
        print(f"  {label:<22}{got:>12,}{flag}")
    if self_conn is not None:
        print(f"  self-connections dropped (logged): {self_conn}   (expected fafb 0, malecns 101)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fafb-edges")
    ap.add_argument("--fafb-annotations", default=ANNOTATIONS_URL)
    ap.add_argument("--fafb-proofread-ids", help="proofread_root_ids_783.npy: restrict FAFB neurons to the release list (139,255)")
    ap.add_argument("--malecns-edges")
    ap.add_argument("--malecns-neurons", help="body-annotations feather (instead of neuPrint)")
    ap.add_argument("--malecns-nt", help="body-neurotransmitters feather, merged into --malecns-neurons")
    ap.add_argument("--only", choices=["fafb", "malecns"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--super-class-map", action="append", default=[], metavar="FROM=TO")
    ap.add_argument("--provisional-maps", action="store_true",
                    help="apply src/provisional_maps.py (super_class values the group has not agreed on)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    extra = dict(m.split("=", 1) for m in a.super_class_map)
    if a.provisional_maps:
        from src.provisional_maps import PROVISIONAL_SUPER_CLASS
        extra = {**PROVISIONAL_SUPER_CLASS, **extra}
    extra = extra or None
    out = {}
    try:
        if a.only != "malecns":
            if not a.fafb_edges:
                ap.error("--fafb-edges is required for FAFB")
            out["fafb"] = fetch_fafb(a.fafb_edges, a.fafb_annotations, a.force, extra, a.fafb_proofread_ids)
        if a.only != "fafb":
            if not a.malecns_edges:
                ap.error("--malecns-edges is required for the male CNS")
            out["malecns"] = fetch_malecns(a.malecns_edges, a.malecns_neurons, a.force, extra, a.malecns_nt)
    except (ValueError, KeyError, RuntimeError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    for name, (neurons, edges, _roi) in out.items():
        step0(name, neurons, edges, None)
    if len(out) == 2:
        assert_interchangeable(out["fafb"][0], out["malecns"][0])
        assert_interchangeable(out["fafb"][1], out["malecns"][1])
        print("\nassert_interchangeable: passed on neurons and edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
