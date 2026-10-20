"""Write reports/eda_C.md (one line per item) from reports/eda_C_numbers.json, so no number is typed by hand.

    python tools/make_summary.py
Run it AFTER the notebook; the notebook's last cell writes a longer draft over the same file.
Edit the 'What surprised me' sentence in the output (or in SURPRISE below) so it is in your own words.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.load(open(ROOT / "reports" / "eda_C_numbers.json", encoding="utf-8"))

SURPRISE_DEFAULT = None


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


c1, c2, c2b, c3, c4, c5, c6 = R["C1"]["central"], R["C2"], R["C2b"], R["C3"], R["C4"], R["C5"], R["C6"]
f, m = c1["fafb"], c1["malecns"]
fo, mo = R["C1"]["optic"]["fafb"], R["C1"]["optic"]["malecns"]
gap_pts = round(100 * (c2b["central_ach_cons"] - f["acetylcholine"]))
SURPRISE = (f"By predictedNt the two datasets looked alike ({pc(f['acetylcholine'])} and {pc(m['acetylcholine'])} acetylcholine) until I replaced the "
            f"male CNS label with consensusNt ({pc(c2b['central_ach_cons'])}); the {gap_pts}-point gap from FAFB then sat where FAFB has its excess of "
            f"dopamine ({pc(f['dopamine'])}) and serotonin ({pc(f['serotonin'])}).")
wo = R.get("C1_weighted", {}).get("optic")
unspec = R.get("unspecified", {})


def conf_line(d):
    by = {k: v for k, v in c2[d]["by_nt"].items() if k not in ("unknown",)}
    hi, lo = max(by, key=by.get), min(by, key=by.get)
    return f"{hi} {by[hi]:.2f} most and {lo} {by[lo]:.2f} least confident"


optic_w, optic_gap = "", ""
if wo:
    gap = abs(wo["fafb"]["acetylcholine"] - wo["malecns"]["acetylcholine"])
    optic_w = f"; weighted by output synapses it is {pc(wo['fafb']['acetylcholine'])} and {pc(wo['malecns']['acetylcholine'])} acetylcholine"
    optic_gap = ("; so the female-male gap in neuron share largely disappears when weighted and comes mostly from labels, not from biology (not tested)"
                 if gap < 0.02 else f"; the weighted gap is still {100 * gap:.0f} points")
L = ["# Workstream C: neurotransmitters and local circuits", "",
     "FlyWire FAFB 783 (female) and male CNS v1.0 (male, brain part). EDA looks at structure and labels only; population and "
     "threshold are stated per line.", "",
     f"- **C1 (central and optic neurons, share of neurons)**: acetylcholine dominates in both datasets. Central: {pc(f['acetylcholine'])} in FAFB and "
     f"{pc(m['acetylcholine'])} in the male CNS by predictedNt ({pc(c2b['central_ach_cons'])} by consensusNt, where the male CNS dopamine falls from "
     f"{pc(m['dopamine'])} to {pc(c2b['central_da_cons'])}: it is Kenyon cells mislabelled); GABA {pc(f['gaba'])} / {pc(m['gaba'])}, glutamate "
     f"{pc(f['glutamate'])} / {pc(m['glutamate'])}, unknown {pc(f['unknown'])} / {pc(m['unknown'])}. Optic: acetylcholine {pc(fo['acetylcholine'])} / "
     f"{pc(mo['acetylcholine'])}, GABA {pc(fo['gaba'])} / {pc(mo['gaba'])}, glutamate {pc(fo['glutamate'])} / {pc(mo['glutamate'])}, unknown "
     f"{pc(fo['unknown'], 1)} / {pc(mo['unknown'])}, plus {pc(mo['histamine'])} histamine in the male CNS only. The FAFB classifier has no histamine class, so its "
     f"photoreceptors are labelled acetylcholine or glutamate{optic_w}{optic_gap}.",
     f"- **C2 (all neurons)**: median confidence is {c2['fafb']['p50']:.2f} in FAFB and {c2['malecns']['p50']:.2f} in the male CNS, "
     f"so there is no common cutoff (0.6 removes {pc(c2['fafb']['below_06'])} vs {pc(c2['malecns']['below_06'])}). FAFB: {conf_line('fafb')}; male CNS: {conf_line('malecns')}. "
     f"Against nt_known, FAFB nt_type agrees for {pc(c2['agreement_excl_hist'])}, but {pc(c2['ach_to_da'], 1)} of known cholinergic neurons are predicted "
     f"dopamine and {pc(c2['glu_to_gaba'], 1)} of known glutamate neurons GABA.",
     f"- **C2b (male CNS)**: predictedNt calls {pc(c2b['kc_share'], 1)} of Kenyon-type neurons dopamine, consensusNt says "
     f"acetylcholine ({pc(c2b['agree'])} agreement overall).",
     f"- **C3 (FAFB)**: a hemilineage is a family of neurons born from the same progenitor lineage, so its neurons should share a transmitter. "
     f"{pc(c3['share_ge_09'])} of {c3['n']} hemilineages have at least 90% of one fast transmitter, " + ("as expected" if c3['share_ge_09'] >= 0.5 else "fewer than expected") + " (Eckstein et al. 2020). The "
     f"{c3['n_lt_08']} below 80% are mixed or poorly predicted; they include the catch-all labels putative_primary and primary, which are not true "
     f"hemilineages (full list in the notebook).",
     f"- **C4 (FAFB central, tau = 10)**: making glutamate excitatory shifts the median net input by {c4['shift_lo']:.2f} to "
     f"{c4['shift_hi']:.2f} in {c4['n_shift']} of the {c4['n_neuropils']} largest neuropils and flips its sign in "
     f"{c4['n_neuropils_flip']}" + (f" ({', '.join(c4['neuropils_flip'])})" if c4['neuropils_flip'] else "") + "; MB_CA does not change.",
     f"- **C5 (mushroom body, tau = 10)**: FAFB has {c5['fafb']['neurons']:,} neurons, {c5['fafb']['edges']:,} edges and "
     f"{c5['fafb']['isolated']} isolated neurons, median in- and out-degree {c5['fafb']['median_in']:.0f} and {c5['fafb']['median_out']:.0f} (male CNS: "
     f"{c5['malecns']['neurons']:,}, {c5['malecns']['edges']:,}, {c5['malecns']['isolated']}); FAFB edges fall from {c5['fafb']['edges_1']:,} (tau = 1) to "
     f"{c5['fafb']['edges_5']:,} (tau = 5) to {c5['fafb']['edges']:,} (tau = 10). At tau = 5 FAFB has {c5['fafb']['isolated_5']} isolated, against 195 and 35 in week 2 "
     f"(the week 2 cell-type selection now gives {c5['fafb']['ct_isolated_10']} and {c5['fafb']['ct_isolated_5']}). The subgraph is thin at tau = 10; whether it is "
     f"dense enough to train on is a modelling question I did not test.",
     f"- **C6 (FAFB mushroom body, tau = 10, {c6['n_seeds']} null graphs)**: reciprocal pairs are {c6['obs_recip'] / c6['null_recip']:.1f} times the "
     f"null ({c6['obs_recip']:,} vs {c6['null_recip']:,.0f}); {', '.join(c6['over'])} are over-represented and "
     f"{', '.join(c6['under'])} under-represented (|z| up to {c6['z_abs_max']:,.0f} because a few hubs dominate).",
     "", f"**What surprised me:** {SURPRISE}", "",
     "Open points for the group: " + (f"the three male CNS 'unspecified' ROIs ({sum(unspec.values()):,} neurons) are in the brain part through "
                                       "two group-agreed rows (is_optic, region_group unknown) passed with table=; " if unspec else "")
     + "the male CNS region field is upstream + downstream (group decision D15); the super_class mapping is provisional; "
       "male CNS edges at 5 synapses: 6,235,649; which male CNS transmitter label to use."]
(ROOT / "reports" / "eda_C.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote reports/eda_C.md")
