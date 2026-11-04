"""Build both caches (corrected tables, eda_shared v0.2.0) and print the Step 0 count check.

    python -m src.build_cache \\
        --fafb-edges data/raw/proofread_connections_783.feather \\
        --fafb-proofread-ids data/raw/proofread_root_ids_783.npy \\
        --fafb-count-pre data/raw/per_neuron_neuropil_count_pre_783.feather \\
        --fafb-count-post data/raw/per_neuron_neuropil_count_post_783.feather \\
        --malecns-syn-partners data/raw/syn-partners-male-cns-v1.0-minconf-0.5.feather \\
        [--malecns-weights data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather] \\
        [--fafb-annotations <tsv or url>] [--only fafb|malecns] [--force]

The male CNS neuron table comes from neuPrint (NEUPRINT_TOKEN in the environment or in a gitignored .env).
After the build, run `python tools/step0.py` for the full Step 0 table to put in the README.
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

EXPECTED = {"fafb neurons": 139_255, "fafb edges >= 5": 2_700_513}


def step0(name: str, neurons: pd.DataFrame, edges: pd.DataFrame) -> None:
    n5 = int((edges["syn_count"] >= 5).sum())
    rows = [(f"{name} neurons", len(neurons)), (f"{name} edges >= 5", n5)]
    print(f"\nStep 0 (short) - {name}")
    for label, got in rows:
        exp = EXPECTED.get(label)
        flag = "" if exp is None else ("  OK" if got == exp else f"  DIFFERENT (expected {exp:,})")
        print(f"  {label:<22}{got:>12,}{flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fafb-edges")
    ap.add_argument("--fafb-annotations", default=ANNOTATIONS_URL)
    ap.add_argument("--fafb-proofread-ids", help="proofread_root_ids_783.npy (Zenodo 10676866): the neuron set, 139,255")
    ap.add_argument("--fafb-count-pre", help="per_neuron_neuropil_count_pre_783.feather (Zenodo 10676866)")
    ap.add_argument("--fafb-count-post", help="per_neuron_neuropil_count_post_783.feather (Zenodo 10676866)")
    ap.add_argument("--malecns-syn-partners", help="syn-partners-male-cns-v1.0-minconf-0.5.feather (6.8 GB)")
    ap.add_argument("--malecns-weights", help="connectome-weights-male-cns-v1.0-minconf-0.5.feather: optional cross-check")
    ap.add_argument("--only", choices=["fafb", "malecns"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    import os
    if os.environ.get("EDA_CACHE_DIR"):          # dry runs on synthetic files only; real runs use config.CACHE_DIR
        config.CACHE_DIR = Path(os.environ["EDA_CACHE_DIR"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    out = {}
    try:
        if a.only != "malecns":
            if not a.fafb_edges:
                ap.error("--fafb-edges is required for FAFB")
            out["fafb"] = fetch_fafb(a.fafb_edges, a.fafb_annotations, a.force, a.fafb_proofread_ids,
                                     a.fafb_count_pre, a.fafb_count_post)
        if a.only != "fafb":
            if not a.malecns_syn_partners:
                ap.error("--malecns-syn-partners is required for the male CNS")
            out["malecns"] = fetch_malecns(a.malecns_syn_partners, a.malecns_weights, a.force)
    except (ValueError, KeyError, RuntimeError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    for name, (neurons, edges, _roi) in out.items():
        step0(name, neurons, edges)
    if len(out) == 2:
        assert_interchangeable(out["fafb"][0], out["malecns"][0])
        assert_interchangeable(out["fafb"][1], out["malecns"][1])
        print("\nassert_interchangeable: passed on neurons and edges")
    print("\nNext: python tools/step0.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
