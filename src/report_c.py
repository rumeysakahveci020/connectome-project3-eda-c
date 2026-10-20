"""Draft of the one-page summary (reports/eda_C.md) from the numbers the notebook computed.

The numbers are exact. The 'what it means' sentences are conditional on them and deliberately
plain; edit them in your own words, and write the surprise line yourself.
"""
from __future__ import annotations

from pathlib import Path


def pc(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.0%}"


def write_report(R: dict, path: Path) -> str:
    L = ["# Workstream C: neurotransmitters and local circuits", "",
         "FlyWire FAFB 783 and male CNS v1.0 (brain part). EDA looks at structure and labels only. "
         "Population and threshold are stated per line.", ""]
    c1 = R["C1"]
    ach = {d: c1["central"][d].get("acetylcholine") for d in ("fafb", "malecns")}
    top = {d: max(c1["central"][d], key=c1["central"][d].get) for d in ("fafb", "malecns")}
    ok = all(top[d] == "acetylcholine" for d in top)
    L.append(f"- **C1 (central, neurons)**: acetylcholine {pc(ach['fafb'])} in FAFB and {pc(ach['malecns'])} in MaleCNS; "
             f"GABA {pc(c1['central']['fafb'].get('gaba'))} / {pc(c1['central']['malecns'].get('gaba'))}, "
             f"glutamate {pc(c1['central']['fafb'].get('glutamate'))} / {pc(c1['central']['malecns'].get('glutamate'))}, "
             f"unknown {pc(c1['central']['fafb'].get('unknown'))} / {pc(c1['central']['malecns'].get('unknown'))}. "
             + ("Acetylcholine is the largest class in both datasets, as expected." if ok else
                f"The largest class is {top['fafb']} in FAFB and {top['malecns']} in MaleCNS, so the datasets disagree "
                "(see C2b before trusting the MaleCNS labels)."))
    c2 = R["C2"]
    L.append(f"- **C2 (all neurons, confidence)**: median confidence {c2['fafb']['p50']:.2f} (FAFB) vs {c2['malecns']['p50']:.2f} "
             f"(MaleCNS); a 0.6 cutoff would remove {pc(c2['fafb']['below_06'])} vs {pc(c2['malecns']['below_06'])}, so one cutoff cannot be applied to both. "
             f"Where nt_known names one transmitter, nt_type agrees for {pc(c2.get('agreement_excl_hist', c2['agreement']))} of FAFB neurons "
             f"({pc(c2['agreement'])} if the histamine neurons, which the classifier cannot predict, are counted). "
             + (f"{pc(c2['ach_to_da'])} of known cholinergic neurons are predicted dopamine and {pc(c2['glu_to_gaba'])} of known glutamate neurons are predicted GABA."
                if "ach_to_da" in c2 else ""))
    if R.get("C2b") is not None:
        L.append(f"- **C2b (male CNS, supplementary)**: predictedNt and consensusNt agree for {pc(R['C2b']['agree'])} of brain neurons; "
                 f"among Kenyon-type neurons predictedNt says {R['C2b']['kc_top']} for {pc(R['C2b']['kc_share'])}. "
                 "This is the check behind the question to the group about which male CNS label to use."
                 + (f" In the central population the male CNS has acetylcholine {pc(R['C2b']['central_ach_pred'])} and dopamine "
                    f"{pc(R['C2b']['central_da_pred'])} by predictedNt, but acetylcholine {pc(R['C2b']['central_ach_cons'])} and dopamine "
                    f"{pc(R['C2b']['central_da_cons'])} by consensusNt, so the C1 shares for the male CNS depend on that choice."
                    if "central_ach_pred" in R["C2b"] else ""))
    c3 = R["C3"]
    L.append(f"- **C3 (FAFB, hemilineages)**: {c3['n']} hemilineages with at least 20 fast-transmitter neurons; "
             f"{pc(c3['share_ge_09'])} have 90% or more of one transmitter; {c3['n_lt_08']} are below 0.8"
             + (f" ({', '.join(c3['lt_08'][:5])})." if c3["lt_08"] else ".")
             + " Most hemilineages share a transmitter, so the column is informative where it is filled."
             if c3["share_ge_09"] >= 0.5 else
             f"- **C3 (FAFB, hemilineages)**: only {pc(c3['share_ge_09'])} of {c3['n']} hemilineages reach 0.9; "
             f"{c3['n_lt_08']} are below 0.8. Many hemilineages are mixed or poorly predicted.")
    c4 = R["C4"]
    L.append(f"- **C4 (FAFB, central, tau = 10)**: " + (f"the glutamate rule shifts the median net index by {c4['shift_lo']:.2f} to {c4['shift_hi']:.2f} in {c4['n_shift']} of the {c4['n_neuropils']} largest neuropils, and " if "n_shift" in c4 else "")
             + f"it changes the sign of the median net index in "
             f"{c4['n_groups_flip']} of {c4['n_groups']} region groups"
             + (f" ({', '.join(c4['groups_flip'])})" if c4["groups_flip"] else "")
             + f" and in {c4['n_neuropils_flip']} of the {c4['n_neuropils']} largest neuropils"
             + (f" ({', '.join(c4['neuropils_flip'][:6])})." if c4["neuropils_flip"] else ".")
             + (" These are the places where the D20 choice matters." if (c4["n_groups_flip"] or c4["n_neuropils_flip"]) else
                " Under this measure the glutamate choice does not flip the median sign anywhere; the sign of individual neurons can still change."))
    c5 = R["C5"]
    L.append(f"- **C5 (mushroom body by region, tau = 10)**: FAFB {c5['fafb']['neurons']:,} neurons, {c5['fafb']['edges']:,} edges, "
             f"density {c5['fafb']['density']:.4f}, {c5['fafb']['isolated']:,} isolated; MaleCNS {c5['malecns']['neurons']:,} neurons, "
             f"{c5['malecns']['edges']:,} edges, density {c5['malecns']['density']:.4f}, {c5['malecns']['isolated']:,} isolated. "
             f"At tau = 5 FAFB has {c5['fafb']['isolated_5']:,} isolated neurons. Week 2 (selected by cell type) found 195 at tau = 10 and 35 at tau = 5.")
    c6 = R["C6"]
    L.append(f"- **C6 (FAFB mushroom body, tau = 10, {c6['n_seeds']} null graphs)**: reciprocal pairs {c6['obs_recip']:,} observed vs "
             f"{c6['null_recip']:.1f} expected (z = {c6['z_recip']:.1f}); "
             f"{len(c6['sig'])} of 14 motifs have |z| > 2"
             + (f": more than expected {', '.join(c6['over']) or 'none'}; fewer than expected {', '.join(c6['under']) or 'none'}." if c6["sig"] else ".")
             + (" These motifs are over- or under-represented beyond what the degrees explain." if c6["sig"] else
                " No motif differs from the degree-preserving null by more than 2 SD, so the degrees explain the local structure.")
             + ("" if c6["n_seeds"] >= 100 else f" Only {c6['n_seeds']} null graphs were run (the plan asks for 100)."))
    L += ["", "**What surprised me:** [write one sentence in your own words]"]
    if R.get("open_points"):
        L += ["", "Open points for the group: " + "; ".join(R["open_points"]) + "."]
    text = "\n".join(L) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text
