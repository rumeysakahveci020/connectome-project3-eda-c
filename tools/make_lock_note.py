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
    return fmt.format(v) if isinstance(v, (int, float)) else str(v)


def below(d):
    try:
        return f"{1 - NUM[d]['tau']['5']['synapses'] / NUM[d]['tau']['1']['synapses']:.0%}"
    except (TypeError, KeyError):
        return "[x]"


pc = lambda *path: g(*path, fmt="{:.0%}")            # noqa: E731
f1 = lambda *path: g(*path, fmt="{:.1f}")            # noqa: E731
F, M = "fafb", "malecns"

ROWS = [
    ("Threshold",
     "Absolute tau = 5 synapses (neuron to neuron, summed over neuropils) as the default; tau = 3 and tau = 10 as sensitivity. The same absolute rule in both datasets. "
     "This is also the mushroom-body threshold.",
     "Kenyon cell to MBON connections are weak per neuron pair, so a strict cutoff removes the output path of the very circuit the project models. The published 10-synapse "
     "threshold (Schlegel et al. 2024) is defined on cell-type edges; ours are neuron-to-neuron, so tau = 10 is not the same measure.",
     f"KC to MBON edges kept vs tau = 1: tau 3 / 5 / 10 = FlyWire {pc(F,'tau','3','kc_to_mbon_kept_vs_tau1')} / {pc(F,'tau','5','kc_to_mbon_kept_vs_tau1')} / {pc(F,'tau','10','kc_to_mbon_kept_vs_tau1')}; "
     f"male CNS {pc(M,'tau','3','kc_to_mbon_kept_vs_tau1')} / {pc(M,'tau','5','kc_to_mbon_kept_vs_tau1')} / {pc(M,'tau','10','kc_to_mbon_kept_vs_tau1')}. Median in-degree at tau = 10: "
     f"{g(F,'tau','10','median_in',fmt='{:.0f}')} and {g(M,'tau','10','median_in',fmt='{:.0f}')} (C5)."),
    ("Population",
     "Mushroom-body subgraph: neurons whose region (shared rule) is MB_CA, MB_PED, MB_VL or MB_ML. Central only; is_optic is not used to select, only to check.",
     "One subgraph defined the same way in both datasets by the shared region rule. The week 2 selection by cell type is kept only as a comparison.",
     f"FlyWire {g(F,'mb_neurons')} and male CNS {g(M,'mb_neurons')} neurons; optic neurons among them: {g(F,'mb_is_optic')} and {g(M,'mb_is_optic')}. "
     f"Cell-type selection {g(F,'mb_neurons_cell_type_selection')} / {g(M,'mb_neurons_cell_type_selection')}, overlap {g(F,'overlap_region_and_cell_type')} / {g(M,'overlap_region_and_cell_type')}."),
    ("Labels",
     "The sign vector comes from nt_type_corrected, never from nt_type. No neuron label is a prediction target. super_class is not a feature. Input and output units are chosen by cell type (see Task).",
     "Week 4 correction 1: the classifiers call Kenyon cells dopamine, which would give the main cell type of the mushroom body no sign.",
     f"Labels in the subgraph (FlyWire): {g(F,'nt_counts')}. Source: {g(F,'source_counts')}. Male CNS: {g(M,'nt_counts')}; {g(M,'source_counts')}."),
    ("Missing values",
     "A neuron with no input or no output at tau = 5 stays in the table with degree 0. In the model, units with no connection at all are removed, the same units in every condition, so unit counts stay matched; "
     "units with input only or output only stay. A neuron with an unknown transmitter keeps its edges and has a free sign.",
     "Isolated units cannot carry or receive a signal in the masked conditions and would only make the dense control different in size. Unknown is an absent label, not a third sign.",
     f"At tau = 5: isolated {g(F,'tau','5','isolated')} (FlyWire) and {g(M,'tau','5','isolated')} (male CNS); no input {g(F,'tau','5','no_input')} / {g(M,'tau','5','no_input')}; no output {g(F,'tau','5','no_output')} / {g(M,'tau','5','no_output')}. "
     f"Unknown label: {pc(F,'unknown_label_share')} / {pc(M,'unknown_label_share')} of units."),
    ("Region ties",
     "The shared rule (alphabetically first name). Reported as a check: how many selected neurons are tied and how many would join or leave under the reverse rule.",
     "Results depend on region only through the selection, so the shared table is used as it is.",
     (f"FlyWire: {g(F,'ties','neurons_with_a_tie')} tied neurons; {g(F,'ties','tied_not_selected_but_could_join')} could join, {g(F,'ties','tied_selected_but_could_leave')} could leave. "
      f"Male CNS: {g(M,'ties','neurons_with_a_tie')} tied; {g(M,'ties','tied_not_selected_but_could_join')} / {g(M,'ties','tied_selected_but_could_leave')}.") if NUM and NUM.get(F, {}).get("ties") else
     "[x] (needs data/raw: run tools/lock_numbers.py on the machine that has the raw files)"),
    ("Split",
     "Folds hold out task examples (stimuli), identical for every condition within a seed; neurons are never held out. Unit ablation is a test-time perturbation with random unit sets drawn per seed, not a fold. At least 10 seeds per condition.",
     "The question compares wiring components on one task, so the score estimates how one fixed network generalises to new stimuli of the same task, not to new neurons or cell types.",
     "Research question and brief (evaluation: at least 10 seeds, identical hyperparameter budget)."),
    ("Cross-dataset method",
     f"Primary: the same absolute tau = 5 in both datasets. Check: a density-matched threshold for the male CNS (the tau with the same edges per neuron as FlyWire at tau = 5).",
     "The brief asks whether the decomposition is consistent across individuals, which needs one protocol. Synapse counts are not on the same scale (about 1.5 times more synapses per connection in the male CNS), so a matched density is the natural check.",
     f"Median synapses per edge at tau = 1: {g('cross_dataset','median_syn_per_edge_tau1',F,fmt='{:.0f}')} vs {g('cross_dataset','median_syn_per_edge_tau1',M,fmt='{:.0f}')}. Edges per neuron at tau = 5: {f1(F,'tau','5','edges_per_neuron')} vs {f1(M,'tau','5','edges_per_neuron')}; "
     f"male CNS tau matching FlyWire's density: {g('cross_dataset','tau_malecns_matching_fafb_tau5_density')}."),
    ("Task and units  [confirm]",
     "One supervised task fixed before any run: K-way classification of noisy binary prototype patterns. Inputs go to the Kenyon-cell units (one input channel per unit); the readout is linear on the MBON units. "
     "[Change this line if your brief or supervisor fixes another task.]",
     "Kenyon cells are the input layer of the mushroom body and MBONs its output, so the assignment follows anatomy and the question, not any result.",
     f"Units: KC {g(F,'kc_neurons')} / {g(M,'kc_neurons')}, MBON {g(F,'mbon_neurons')} / {g(M,'mbon_neurons')} (FlyWire / male CNS)."),
    ("Sign rule",
     "Glutamate is inhibitory by default; excitatory glutamate is the sensitivity condition. Dopamine, serotonin, octopamine and unknown units have a free sign in the sign-constrained condition (they are modulators, no fast sign).",
     "Glutamate acts through GluCl-alpha channels as an inhibitory transmitter in the Drosophila olfactory system (Liu and Wilson 2013), but not everywhere, so the other rule stays as a check.",
     f"C4 on the corrected labels: the glutamate rule shifts the median net input by at least 0.25 in {g('c4_corrected','shifted_025')} of {g('c4_corrected','neuropils')} neuropils and flips its sign in {g('c4_corrected','flip')}. "
     f"Units with a free sign: {g(F,'signs','rule1_glutamate_inhibitory','free')} / {g(M,'signs','rule1_glutamate_inhibitory','free')}; units + / - under rule 1: "
     f"{g(F,'signs','rule1_glutamate_inhibitory','plus')} / {g(F,'signs','rule1_glutamate_inhibitory','minus')} (FlyWire)."),
    ("Mask below the threshold",
     "A connection below tau is 0 (not in the mask); there is no third 'unknown' state. Taus 3 and 10 show how much this matters.",
     "The ladder compares masks, so each pair is either allowed or forbidden; an unknown state would add a condition nobody asked for.",
     f"Share of synapses in edges below tau = 5: FlyWire {below(F)}, male CNS {below(M)}."),
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
    st.font.name, st.font.size = "Calibri", Pt(7.5)
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
    doc.add_paragraph("Sources: Schlegel et al., Nature 2024; Liu and Wilson, PNAS 2013; Eckstein et al., Cell 2024; Week 4 Shared Data Corrections; Project 3 brief.").runs[0].font.size = Pt(6.5)
    path.parent.mkdir(exist_ok=True)
    doc.save(path)


if __name__ == "__main__":
    out = ROOT / "reports" / ("lock_note_P3_DRAFT.docx" if DRAFT else "lock_note_P3.docx")
    build(out)
    print("wrote", out, "(draft: numbers are [x])" if DRAFT else "")
