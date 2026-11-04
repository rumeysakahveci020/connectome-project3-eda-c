"""Step 0 on the corrected tables: the numbers the Shared Data Corrections ask every repository to report.

    python tools/step0.py            # prints the table and writes reports/step0_C.md (paste it into the README)

The first repository to finish posts its table to the group; every other one must match it exactly. A mismatch is a fetch
problem, so fix the fetch, not the number. Reference values are the ones in the Corrections document.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from eda_shared.corrections import kenyon_cell_types, nt_correction_summary  # noqa: E402
from src import eda_utils as U  # noqa: E402

REF_SUPER = {"optic": 77537, "central": 32383, "sensory": 16904, "visual_projection": 8038, "ascending": 1750,
             "descending": 1303, "sensory_ascending": 612, "visual_centrifugal": 524, "motor": 110, "endocrine": 80, "unknown": 14}
REF = {"fafb_neurons": 139255, "fafb_kept": 139241, "fafb_added": 14, "fafb_dropped": 7, "fafb_edges_ge5": 2700513,
       "fafb_self": 0, "malecns_self": 101, "known_nt_rows": 77169, "known_nt_changed": 18513, "fafb_dopamine_after": 638,
       "fafb_dopamine_before": 5909, "kc_fafb": 5177}


def report(name: str) -> dict:
    p = config.CACHE_DIR / f"step0_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    U._cache_dir()
    out, bad = [], []

    def row(label, got, ref=None, note=""):
        flag = ""
        if ref is not None:
            flag = "  OK" if got == ref else f"  DIFFERENT (reference {ref:,})"
            if got != ref:
                bad.append(label)
        g = f"{got:,}" if isinstance(got, (int,)) else str(got)
        out.append(f"| {label} | {g}{flag} | {note} |")

    f_n, f_e = U.load("fafb")
    m_n, m_e = U.load("malecns")
    fr, mr = report("fafb"), report("malecns")

    row("FAFB neurons", len(f_n), REF["fafb_neurons"], "the 783 proofread list")
    row("FAFB annotation rows kept", len(f_n) - fr.get("added_unknown", 0), REF["fafb_kept"])
    row("FAFB proofread ids added (unknown labels)", fr.get("added_unknown", "n/a"), REF["fafb_added"])
    row("FAFB annotation rows dropped", fr.get("dropped_not_in_release", "n/a"), REF["fafb_dropped"])
    sc = f_n["super_class"].astype(str).value_counts().to_dict()
    for k, v in REF_SUPER.items():
        row(f"FAFB super_class {k}", sc.get(k, 0), v)
    row("FAFB edges at tau = 1", len(f_e))
    row("FAFB edges at >= 5 synapses", int((f_e["syn_count"] >= 5).sum()), REF["fafb_edges_ge5"], "published Buhmann count")
    reg = f_n["region"].astype(str).value_counts()
    row("FAFB neurons with region unknown", int(reg.get("unknown", 0)))
    row("FAFB region ties (alphabetical rule)", fr.get("region_ties", "n/a"))
    row("FAFB regions (neurons per region, top 8)", "; ".join(f"{k} {v:,}" for k, v in reg.head(8).items()))
    row("FAFB self-connections dropped", fr.get("self_connections", "n/a"), REF["fafb_self"])

    row("MaleCNS Traced neurons (before the crop)", mr.get("neurons_traced", "n/a"), 165122)
    row("MaleCNS brain part: neurons kept", len(m_n))
    row("MaleCNS brain part: neurons dropped", mr.get("neurons_dropped", "n/a"))
    row("MaleCNS ascending kept", mr.get("ascending_kept", "n/a"))
    row("MaleCNS sensory_ascending kept", mr.get("sensory_ascending_kept", "n/a"))
    row("MaleCNS brain edges at tau = 1", len(m_e))
    row("MaleCNS brain edges at >= 5 synapses", int((m_e["syn_count"] >= 5).sum()))
    row("MaleCNS synapses in brain / nerve cord / no location",
        f"{mr.get('edgeroi_syn_brain', 'n/a'):,} / {mr.get('edgeroi_syn_nerve_cord', 'n/a'):,} / {mr.get('edgeroi_syn_no_location', 'n/a'):,}"
        if mr else "n/a")
    row("MaleCNS self-connections dropped", mr.get("self_connections", "n/a"), REF["malecns_self"])
    wc = mr.get("weights_check")
    if wc:
        row("MaleCNS per-neuropil counts equal connectome-weights", str(wc.get("agree")), None, f"{wc}")

    for nm, n in (("FAFB", f_n), ("MaleCNS", m_n)):
        c = n[(~n["is_optic"]) & (n["region"].astype(str) != "unknown")]
        row(f"{nm} central population (is_optic False, region known)", len(c))

    summ = nt_correction_summary(pd.concat([f_n, m_n], ignore_index=True))
    for _, r in summ.iterrows():
        row(f"Transmitter correction {r['dataset']} / {r['nt_correction_source']}", f"{int(r['rows']):,} rows, {int(r['changed']):,} change")
    dop = f_n["nt_type"].astype(str).eq("dopamine").sum(), f_n["nt_type_corrected"].astype(str).eq("dopamine").sum()
    row("FAFB dopamine neurons before -> after", f"{dop[0]:,} -> {dop[1]:,}", None, "reference 5,909 -> 638")
    kc = kenyon_cell_types(pd.concat([f_n, m_n], ignore_index=True))
    row("Kenyon-cell types (print once, confirm only Kenyon cells)", "; ".join(f"{r.dataset}:{r.cell_type} {r.neurons:,}" for r in kc.itertuples()))

    md = ["# Step 0 on the corrected tables (eda_shared v0.2.0)", "", "| Quantity | Value | Note |", "| --- | --- | --- |", *out, ""]
    text = "\n".join(md)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "step0_C.md").write_text(text, encoding="utf-8")
    print("\nwrote reports/step0_C.md")
    if bad:
        print("\nDIFFERENT from the reference:", bad, "\n-> a mismatch is a fetch problem: fix the fetch, not the number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
