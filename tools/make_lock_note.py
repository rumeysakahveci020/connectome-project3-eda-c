"""Writes the one-page Project 3 lock note (Word) from reports/lock_numbers.json, so no number is typed by hand.

    python tools/lock_numbers.py        # first: the numbers, on the corrected tables
    python tools/make_lock_note.py      # reports/lock_note_P3.docx   (without the numbers file: a DRAFT with [..] slots)

The CHOICES are proposals written from the project brief, the EDA and the literature; the note is yours: read every line, change what
you do not agree with, and keep the rule of the assignment: no choice may rest on any model's output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from docx import Document  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

p = ROOT / "reports" / "lock_numbers.json"
NUM = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
DRAFT = NUM is None
_pl = ROOT / "reports" / "mb_layers.json"
LAY = json.loads(_pl.read_text(encoding="utf-8")) if _pl.exists() else None


def g(*path, fmt="{:,}", default="[x]"):
    """Pick a number out of lock_numbers.json; [x] in a draft."""
    if NUM is None:
        return default
    v = NUM
    for k in path:
        try:
            v = v[k]
        except (KeyError, TypeError):
            return default
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        return ", ".join(f"{k} {x:,}" if isinstance(x, int) else f"{k} {x}" for k, x in v.items())
    return fmt.format(v) if isinstance(v, (int, float)) else str(v)


def below(d):
    """Share of synapses (or, if the numbers file has no synapse totals, of edges) in edges below tau = 5, with its unit."""
    try:
        t = NUM[d]["tau"]
        return f"{1 - t['5']['synapses'] / t['1']['synapses']:.0%} of synapses"
    except (TypeError, KeyError, ZeroDivisionError):
        try:
            return f"{1 - NUM[d]['tau']['5']['edges'] / NUM[d]['tau']['1']['edges']:.0%} of edges"
        except (TypeError, KeyError, ZeroDivisionError):
            return "[x]"


def ratio5():
    try:
        return f"{NUM[M]['tau']['5']['edges_per_neuron'] / NUM[F]['tau']['5']['edges_per_neuron']:.1f}"
    except (TypeError, KeyError, ZeroDivisionError):
        return "[x]"


def paths(d):
    try:
        p = NUM[d]["design"]["5"]
        lengths = ", ".join(f"{v} at {k}" for k, v in p["path_counts"].items())
        return f"{p['mbon_reachable']} of {p['mbon_total']} MBONs reachable from a Kenyon cell, shortest paths: {lengths}"
    except (TypeError, KeyError):
        return "[x]"


def pmatch(d):
    try:
        p = NUM[d]["design"]["5"]
        return f"E = {p['edges']:,}, N = {p['units']:,}, N^2 = {p['dense_weights']:,}, H = {p['matched_dense_width']}"
    except (TypeError, KeyError):
        return "[x]"


def union():
    return bool(NUM) and NUM.get("selection") == "union"


pc = lambda *path: g(*path, fmt="{:.0%}")            # noqa: E731
f1 = lambda *path: g(*path, fmt="{:.1f}")            # noqa: E731
F, M = "fafb", "malecns"

ROWS = [
    ("Threshold",
     "Absolute tau = 5 synapses (neuron to neuron, summed over neuropils) as the default; tau = 3 and tau = 10 as sensitivity. The same absolute rule in both datasets. "
     "This is also the mushroom-body threshold.",
     "Five synapses is the neuron-level threshold used for FlyWire (Dorkenwald et al. 2024); it limits low-confidence connections. Kenyon cell to MBON connections are weak per neuron pair, "
     "so a stricter cutoff removes the output path of the very circuit the project models, while a looser one adds weak connections: tau = 3 and tau = 10 bracket the choice. The published 10-synapse "
     "threshold (Schlegel et al. 2024) is defined on cell-type edges; ours are neuron-to-neuron, so tau = 10 is not the same measure.",
     f"KC to MBON edges kept vs tau = 1: tau 3 / 5 / 10 = FlyWire {pc(F,'tau','3','kc_to_mbon_kept_vs_tau1')} / {pc(F,'tau','5','kc_to_mbon_kept_vs_tau1')} / {pc(F,'tau','10','kc_to_mbon_kept_vs_tau1')}; "
     f"male CNS {pc(M,'tau','3','kc_to_mbon_kept_vs_tau1')} / {pc(M,'tau','5','kc_to_mbon_kept_vs_tau1')} / {pc(M,'tau','10','kc_to_mbon_kept_vs_tau1')}. Median in-degree at tau = 10: "
     f"{g(F,'tau','10','median_in',fmt='{:.0f}')} and {g(M,'tau','10','median_in',fmt='{:.0f}')} (C5). tau = 5 keeps the smaller share in FlyWire, so tau = 3 is the first sensitivity."),
    ("Population",
     ("Mushroom-body subgraph: neurons whose region (shared rule) is MB_CA, MB_PED, MB_VL or MB_ML, plus every neuron of a mushroom-body cell type (Kenyon cells, MBONs, dopaminergic and other MB types). "
      "Central only; is_optic is not used to select, only to check.") if union() else
     ("Mushroom-body subgraph: neurons whose region (shared rule) is MB_CA, MB_PED, MB_VL or MB_ML. Central only; is_optic is not used to select, only to check. "
      "Kenyon cells and MBONs that the region rule drops stay outside; the count is given as evidence."),
     ("The shared region rule counts output synapses too, so a mushroom-body neuron that sends most of its signal elsewhere gets another neuropil as its region. "
      "Input and output layers must be complete, so the rule is widened with the cell types.") if union() else
     "One subgraph defined the same way in both datasets by the shared region rule. The week 2 selection by cell type is kept only as a comparison.",
     (f"Region rule alone: Kenyon cells {g(F,'kc_in_region_rule')} of {g(F,'kc_all')}, MBONs {g(F,'mbon_in_region_rule')} of {g(F,'mbon_all')} (FlyWire); "
      f"{g(M,'kc_in_region_rule')} of {g(M,'kc_all')} and {g(M,'mbon_in_region_rule')} of {g(M,'mbon_all')} (male CNS). Subgraph used: FlyWire {g(F,'mb_neurons')}, male CNS {g(M,'mb_neurons')} neurons; "
      f"optic neurons among them {g(F,'mb_is_optic')} and {g(M,'mb_is_optic')}."
      + (f" The dropped MBONs send {LAY['fafb']['mbon_dropped_outputs_in_mb']:.0%} (FlyWire) and {LAY['malecns']['mbon_dropped_outputs_in_mb']:.0%} (male CNS) of their output sites (median) into the mushroom body; "
         f"the {LAY['fafb']['dropped_kc']} dropped FlyWire Kenyon cells sit mostly in {', '.join(list(LAY['fafb']['dropped_regions'].get('KC', {}))[:2])}." if LAY else ""))),
    ("Labels",
     "The sign vector comes from nt_type_corrected, never from nt_type. No neuron label is a prediction target. super_class is not used, so no levels are merged. Input and output units are chosen by cell type (see Input and output units).",
     "Week 4 correction 1: the classifiers call Kenyon cells dopamine, which would give the main cell type of the mushroom body no sign.",
     f"Labels in the subgraph (FlyWire): {g(F,'nt_counts')}. Source: {g(F,'source_counts')}. Male CNS: {g(M,'nt_counts')}; {g(M,'source_counts')}. "
     f"Type mix (types with at least 5 neurons, top transmitter below 90%): FlyWire {g(F,'type_mix','mixed_types')} of {g(F,'type_mix','types')}, "
     f"male CNS {g(M,'type_mix','mixed_types')} of {g(M,'type_mix','types')} ({g(M,'type_mix','neurons_in_mixed_types')} neurons)."),
    ("Missing values",
     "A neuron with no input or no output at tau = 5 stays in the table with degree 0. In the model, units with no connection at all are removed, the same units in every condition, so unit counts stay matched; "
     "units with input only or output only stay. A neuron with an unknown transmitter keeps its edges and has a free sign.",
     "Isolated units cannot carry or receive a signal in the masked conditions and would only make the dense control different in size. Unknown is an absent label, not a third sign.",
     f"At tau = 5: isolated {g(F,'tau','5','isolated')} (FlyWire) and {g(M,'tau','5','isolated')} (male CNS); no input {g(F,'tau','5','no_input')} / {g(M,'tau','5','no_input')}; no output {g(F,'tau','5','no_output')} / {g(M,'tau','5','no_output')}. "
     f"Unknown label: {g(F,'unknown_label_share',fmt='{:.1%}')} / {g(M,'unknown_label_share',fmt='{:.1%}')} of units."),
    ("Region ties",
     "The shared rule (alphabetically first name). Reported as a check: how many selected neurons are tied and how many would join or leave under the reverse rule.",
     "Results depend on region only through the selection, so the shared table is used as it is.",
     (f"FlyWire: {g(F,'ties','neurons_with_a_tie')} tied neurons; {g(F,'ties','tied_not_selected_but_could_join')} could join, {g(F,'ties','tied_selected_but_could_leave')} could leave. "
      f"Male CNS: {g(M,'ties','neurons_with_a_tie')} tied; {g(M,'ties','tied_not_selected_but_could_join')} / {g(M,'ties','tied_selected_but_could_leave')}.") if NUM and NUM.get(F, {}).get("ties") else
     "[x] (needs data/raw: run tools/lock_numbers.py on the machine that has the raw files)"),
    ("Split",
     "Training uses all n noisy copies of each of the P patterns. The validation set (1,000 samples) and the test set (2,000 samples) are new noisy copies of the same P patterns, generated once per seed and identical for every condition; no pattern is removed from training. Neurons are never held out. Unit ablation is a test-time perturbation with random unit sets drawn per seed. At least 10 seeds per condition.",
     "Labels are random, so a network can learn a pattern's label only by seeing that pattern in training; the task measures recognising new noisy copies of seen patterns. A validation set carved out of the training copies would leave some patterns unseen (all of them at n = 1), which would break learning-rate selection and early stopping.",
     "Research question and brief (evaluation: at least 10 seeds, identical hyperparameter budget)."),
    ("Cross-dataset method",
     "Primary: the same absolute tau = 5 in both datasets. Check: a density-matched threshold for the male CNS (the tau whose edges per neuron equal FlyWire's at tau = 5, found by scanning). "
     "The scaling used by Berg et al. (male weights scaled down, a separate noise threshold per dataset) is not applied.",
     "The brief asks whether the decomposition is consistent across individuals, which needs one protocol. The male CNS carries about 1.5 times the synapses of FAFB (a Janelia preprint, quoted in the Corrections), "
     "so the same tau is not the same stringency; a matched density is the natural check.",
     f"Edges per neuron at tau = 5: {f1(F,'tau','5','edges_per_neuron')} (FlyWire) vs {f1(M,'tau','5','edges_per_neuron')} (male CNS), a factor of {ratio5()}. "
     f"Male CNS tau matching FlyWire's tau = 5 density: {g('cross_dataset','tau_malecns_matching_fafb_tau5_density')}."),
    ("Training data",
     "Synthetic pattern task (Litwin-Kumar et al. 2017). P random binary patterns of dimension d = 50 and density f = 0.5, each with a random label +1 or -1. A sample is a noisy copy of a pattern: every bit flips with probability 0.1. "
     "P in {10, 20, 40, 80, 160, 320}; n training copies per pattern in {1, 3, 10, 30, 100}; a fixed validation set of 1,000 and a fixed test set of 2,000 new noisy copies of the same P patterns. Data are generated from the seed; within a seed every condition sees the same training, validation and test data.",
     "The standard mushroom-body task. A linear readout can separate only about twice as many random patterns as it has units (Cover 1965): with the readout on the MBON units this is about 2 x 96 = 192 patterns in FlyWire (2 x 97 = 194 in the male CNS), so the grid up to 320 passes the limit and spans easy to impossible without choosing P from results.",
     "Litwin-Kumar et al., Neuron 2017 (d, f, noise); Cover 1965; the Project 3 brief (learning curves, sample efficiency)."),
    ("Input and output units",
     "The input has 50 channels; each Kenyon cell is wired to 7 randomly chosen channels (Caron et al. 2013). These input weights are fixed, drawn once per seed, identical in every condition and in both datasets. "
     "The readout is a linear layer on the MBON units, not masked and without sign constraint.",
     f"One channel per Kenyon cell would give inputs of {g(F,'kc_neurons')} and {g(M,'kc_neurons')} dimensions, two different problems; 50 channels give the same problem in both datasets. Kenyon cells are the input layer of the mushroom body and MBONs its output.",
     f"Units in the subgraph: Kenyon cells {g(F,'kc_neurons')} / {g(M,'kc_neurons')}, MBONs {g(F,'mbon_neurons')} / {g(M,'mbon_neurons')} (FlyWire / male CNS). Caron et al., Nature 2013."),
    ("Trainable weights and loss",
     "Trained: every weight on an edge of the mask (Kenyon cell to MBON and all recurrent edges), a bias per unit and the readout. Not trained: the input mapping. Loss: binary cross-entropy for the +1 / -1 label.",
     "The mask decides which recurrent weights exist; everything else is identical in all conditions.",
     "Project 3 brief."),
    ("Optimiser and budget",
     "Adam, then a projection after every step: apply the mask and clip each weight to the sign of its presynaptic unit (units with a free sign are not clipped). Learning rate from the grid {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, chosen once per (dataset, condition) pair, on the validation loss of seed 0 at P = 40 and n = 10; the same grid and the same five runs for every pair, never the test set. The chosen rate is then fixed for all P, n and seeds. "
     "200 epochs with early stopping on the validation loss (patience 20).",
     "The brief asks for projected gradient descent and an identical hyperparameter budget; unequal tuning is the most common way this comparison breaks.",
     "Project 3 brief; Lock Note assignment (no choice may rest on a model output)."),
    ("Architecture and initialisation",
     f"Recurrent network on the masked graph: h_0 = 0; h_t = ReLU(W h_(t-1) + W_in x + b), t = 1..T; output = readout of h_T on the MBON units (W: one weight per edge of the mask; W_in: fixed, 7 input connections per Kenyon cell, each of weight 1/sqrt(7); x: the 50-channel pattern). "
     f"T = the longest shortest path from a Kenyon cell to a reachable MBON at tau = 5, plus one (longest shortest path: FlyWire {g(F,'design','5','path_max')}, male CNS {g(M,'design','5','path_max')}; the larger is used for both). "
     "Initial weight magnitudes |W_ij| ~ U(0, 2 / sqrt(k_i)), k_i the in-degree of unit i in the mask. Initial signs: every unit gets one initial sign, drawn once per seed (a unit with a sign keeps it; a unit with a free sign, such as dopamine or unknown, gets a fair coin per unit, not per weight); the initial sign of a weight is the initial sign of its presynaptic unit. "
     "This vector is the same in every condition, with and without the sign constraint; the constraint acts only through the projection. In the dense control, which has no neurons, unit signs are drawn from the empirical sign distribution of the true graph. One rule for all conditions and both datasets.",
     "T follows the graph, not a result. A common initialisation keeps a difference that comes from a bad start (almost all signed units are positive) from being read as a difference of architecture. With the input applied at every step and h_0 = 0, input units are active at step 1 and a path of length L reaches its MBON at step L + 1, so T = longest shortest path + 1 lets every reachable MBON be read.",
     f"FlyWire: {paths('fafb')}. Male CNS: {paths('malecns')}. Units with a positive sign {g(F,'signs','rule1_glutamate_inhibitory','plus')} and a negative sign {g(F,'signs','rule1_glutamate_inhibitory','minus')} (FlyWire)."),
    ("Parameter matching",
     "The three sparse conditions share the number of masked recurrent weights E of the true mask. The dense condition uses H = round(sqrt(E)) units so that its H x H recurrent weights equal E too. Here the parameters match and the units do not: the dense network has H units, the sparse ones N. "
     "Dense input and output: the last n_out = number of MBONs of the true graph units are read out (the same readout size as every other condition); the other H - n_out units receive the 50-channel input through the same fixed 7-channel rule. The fixed input mapping and the biases are left out of the match.",
     "The brief asks for equal parameter counts; a dense network on all N units would have N^2 weights, far more than the mask. The dense control has no neurons, so its input and output roles are assigned by position.",
     f"At tau = 5, FlyWire: {pmatch('fafb')}. Male CNS: {pmatch('malecns')}."),
    ("Implementation details",
     "The input drives the input units at every step (the update equation above); the readout reads the MBON units at the last step. Batch size 64; "
     "readout weights start from N(0, 1/n_out), biases at 0. The random-sparse mask draws E distinct pairs uniformly over all units.",
     "The brief does not fix these details; writing them down before any run keeps them from being tuned to a result.",
     "src/model and tests/test_model.py (the training code, tested against finite differences)."),
    ("Sign rule",
     "Glutamate is inhibitory by default; excitatory glutamate is the sensitivity condition. Dopamine, serotonin, octopamine and unknown units have a free sign in the sign-constrained condition (they are modulators, no fast sign).",
     "Glutamate acts through GluCl-alpha channels as an inhibitory transmitter in the Drosophila olfactory system (Liu and Wilson 2013), but not everywhere, so the other rule stays as a check.",
     f"In this network the glutamate rule changes the sign of {g(F,'glutamate_units_changing_sign')} units in FlyWire and {g(M,'glutamate_units_changing_sign')} in the male CNS (units labelled glutamate; inhibitory under rule 1, excitatory under rule 2). "
     f"Units with a free sign: {g(F,'signs','rule1_glutamate_inhibitory','free')} / {g(M,'signs','rule1_glutamate_inhibitory','free')}; units + / - under rule 1: "
     f"{g(F,'signs','rule1_glutamate_inhibitory','plus')} / {g(F,'signs','rule1_glutamate_inhibitory','minus')} (FlyWire), {g(M,'signs','rule1_glutamate_inhibitory','plus')} / {g(M,'signs','rule1_glutamate_inhibitory','minus')} (male CNS)."),
    ("Mask below the threshold",
     "A connection below tau is 0 (not in the mask); there is no third 'unknown' state. Taus 3 and 10 show how much this matters.",
     "The ladder compares masks, so each pair is either allowed or forbidden; an unknown state would add a condition nobody asked for.",
     f"Edges below tau = 5 hold: FlyWire {below(F)} (tau = 1 as reference), male CNS {below(M)}."),
]


def build(path: Path) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(1.1))
    sec.top_margin, sec.bottom_margin = Cm(0.9), Cm(0.8)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(7)
    st.paragraph_format.space_after = Pt(0)
    t = doc.add_paragraph()
    r = t.add_run("Lock note: Project 3, Decomposing the biological prior (sparsity, degree structure or sign?)")
    r.bold, r.font.size = True, Pt(11)
    s = doc.add_paragraph("Rumeysa. Numbers recomputed on the corrected tables (eda_shared v0.2.0, nt_type_corrected); every choice rests on the EDA, the literature or the research question, "
                          "none on a model's output." + ("  DRAFT: [x] = number to fill after tools/lock_numbers.py." if DRAFT else ""))
    s.runs[0].italic = True
    tb = doc.add_table(rows=1, cols=4)
    tb.style = "Table Grid"
    widths = [Cm(2.4), Cm(9.0), Cm(8.2), Cm(8.0)]
    for i, h in enumerate(("Item", "Choice", "Reason", "Evidence (number or source)")):
        c = tb.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h); run.bold = True
    for item, choice, reason, ev in ROWS:
        cells = tb.add_row().cells
        for i, txt in enumerate((item, choice, reason, ev)):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(txt)
            run.bold = i == 0
    for row in tb.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = w
    doc.add_paragraph("Sources: Schlegel et al., Nature 2024; Liu and Wilson, PNAS 2013; Eckstein et al., Cell 2024; Dorkenwald et al., Nature 2024; Berg et al., Cell 2026; Litwin-Kumar et al., Neuron 2017; Caron et al., Nature 2013; Cover 1965; Week 4 Shared Data Corrections; Project 3 brief.").runs[0].font.size = Pt(6.5)
    path.parent.mkdir(exist_ok=True)
    doc.save(path)


if __name__ == "__main__":
    out = ROOT / "reports" / ("lock_note_P3_DRAFT.docx" if DRAFT else "lock_note_P3.docx")
    build(out)
    print("wrote", out, "(draft: numbers are [x])" if DRAFT else "")
