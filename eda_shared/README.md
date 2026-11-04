# Shared files — v0.2.0

This release puts *Week 4 — Shared Data Corrections* into code. It replaces v0.1.1-eda. Copy it unchanged:

- replace the `eda_shared/` folder in the root of your repository with this one;
- replace `config.py` with the `config.py` shipped next to this folder (v0.2.0).

Do not edit either file locally. Report a problem to the group; the fix comes back as a new version for everyone. What changed is in `CHANGELOG.md`.

Notebooks run from `notebooks/`, so their first cell puts the repository root on the import path:

```python
import sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
```

After copying, run the tests once from the repository root: `pytest eda_shared/tests` (49 tests). One of them checks that `config.py` is v0.2.0.

## Files

| File | What it gives you |
| --- | --- |
| `corrections.py` | One function per correction: `add_nt_type_corrected`, `map_super_class`, `align_to_proofread`, `region_from_counts` (with `fafb_neuropil_counts` and `malecns_neuropil_counts`), `malecns_brain_part`; plus `kenyon_cell_types` and `nt_correction_summary` for the Step 0 report |
| `eda_helpers.py` | `add_provisional_flags`, `basic_features`, `relative_weight`, and the MaleCNS nerve-cord list `MALECNS_NERVE_CORD` |
| `neuropils_provisional.csv` | Each neuropil with `is_optic` and `region_group`, including `Optic-unspecified` and `CentralBrain-unspecified` |
| `randomiser.py` | `randomise_edges` (degree-preserving, seeded), `reciprocity`. Unchanged |
| `tests/` | 15 tests from v0.1.1 and 34 for the corrections, including a MaleCNS brain part worked by hand |

## Inputs you need

| Dataset | File | Used for |
| --- | --- | --- |
| FAFB | `Supplemental_file1_neuron_annotations.tsv`, annotation release v3.2.0 | Labels |
| FAFB | `proofread_root_ids_783.npy` (Zenodo record 10676866) | The neuron set |
| FAFB | `per_neuron_neuropil_count_pre_783.feather` and `per_neuron_neuropil_count_post_783.feather` (same record) | Region |
| FAFB | Edges and edgeroi from the 783 proofread connections, as before | Edges |
| MaleCNS | neuPrint `male-cns:v1.0`: `fetch_neurons` (neuron table and per-ROI table) and `fetch_primary_rois` | Labels, region |
| MaleCNS | edgeroi cached at 1 synapse, one row per primary neuropil | Brain edges |

**MaleCNS region source.** Use neuPrint's per-ROI `upstream` and `downstream`, primary ROIs only. The Janelia synapse-partner file gives about 1% different counts (750 against 761 neurons with an unspecified region), so it cannot be used for the shared table: Step 0 must match exactly between repositories.

## Building the corrected tables

Do this once, in your fetch code, and cache the result. Then rerun Step 0.

**FAFB**

```python
import numpy as np
from eda_shared.corrections import (map_super_class, align_to_proofread, fafb_neuropil_counts,
                                    region_from_counts, add_nt_type_corrected)

# neurons: one row per row of the annotation file, in the same order as `ann`
sc = map_super_class(ann["super_class"], "fafb")                       # correction 2
neurons["super_class"] = sc["super_class"].array
neurons["super_class_source"] = sc["super_class_source"].array
neurons, rep = align_to_proofread(neurons, np.load("proofread_root_ids_783.npy"))   # 3
counts = fafb_neuropil_counts(pre, post)                               # the two count files
reg = region_from_counts(neurons["neuron_id"], counts)                 # correction 6
neurons["region"] = neurons["neuron_id"].map(reg.set_index("neuron_id")["region"]).astype("string")
neurons = add_nt_type_corrected(neurons)                               # correction 1
```

- `align_to_proofread` keeps the 139,241 annotation rows whose id is in the 783 proofread list, drops the 7 that are not (newer root ids, no edges in 783), and adds the 14 proofread ids that have no annotation row, with `unknown` labels. Total 139,255, the published count. Its report gives the split for Step 0.
- Keep this order. `align_to_proofread` refuses a column it cannot fill with `unknown` or a missing value (such as `nt_correction_source` or a boolean `is_optic`), so build the neuron set before adding those columns.
- Compute `region` for all 139,255 neurons, including the 14 added ones: they have synapses, so they get a real region.

**MaleCNS**

```python
from eda_shared.corrections import (map_super_class, add_nt_type_corrected,
                                    malecns_neuropil_counts, malecns_brain_part)
from eda_shared.eda_helpers import relative_weight

sc = map_super_class(raw_superclass, "malecns")                        # correction 2
neurons["super_class"] = sc["super_class"].array
neurons["super_class_source"] = sc["super_class_source"].array
neurons = add_nt_type_corrected(neurons)                               # correction 1

roi_counts = malecns_neuropil_counts(roi_df[roi_df["roi"].isin(primary_rois)])
neurons_b, edges_b, edgeroi_b, report = malecns_brain_part(neurons, edgeroi, roi_counts)  # 4, 6
edges_b["frac"] = relative_weight(edges_b)                             # correction 5
```

`malecns_brain_part` returns:

- `neurons_b`: the kept neurons, with `region` recomputed by the shared rule over brain neuropils;
- `edges_b`: brain edges summed over neuropils, `min_syn_fetched = 1`. This is the MaleCNS edge table for `basic_features`, `relative_weight` and every threshold;
- `edgeroi_b`: the brain rows of edgeroi;
- `report`: the Step 0 counts (neurons kept and dropped, ascending neurons kept, brain edges at τ = 1 and ≥ 5, synapses by location).

Rules (correction 4):

- **Neurons:** kept if they have at least one connection, with any partner, in a brain neuropil.
- **Brain neuropils:** every neuropil in the table, including the two unspecified labels. `MALECNS_NERVE_CORD` is not brain.
- **No location:** a synapse outside every primary ROI (`<unspecified>`, `NotPrimary`) counts as neither brain nor nerve cord.
- **Memory:** on a 29M-row edgeroi the call peaks near 5 GB. Store `neuropil` as a categorical and `syn_count` as int32 when you cache edgeroi.

## The region rule (both datasets)

`region_from_counts` implements it:

- **Counts:** inputs plus outputs per neuron and neuropil, **over all partners**, proofread or not. This is the Week 2 rule ("from all synapses"). In FAFB a neuron's outputs mostly go to non-proofread fragments (only 44.7% of postsynapses are attached to proofread neurons, Dorkenwald et al. 2024), so a count from edgeroi would weight inputs and is not used.
- **Candidates:** brain neuropils only. FAFB `UNASGD` / `None` and MaleCNS `<unspecified>` / `NotPrimary` are never a region, and neither is a nerve-cord neuropil.
- **Ties:** the alphabetically first name, in string order whatever the column's dtype. The tie rule is fixed so the shared table is identical everywhere; a project that needs another rule recomputes it from the counts in its own code and says so.
- **No candidate:** `unknown`.

## The corrected columns

| Column | Rule |
| --- | --- |
| `nt_type_corrected` | FAFB: `nt_known` where it names exactly one transmitter, otherwise `nt_type`. MaleCNS: `acetylcholine` for Kenyon-cell types (`cell_type` starting with `KC`), otherwise `nt_type`. **Use this column in any analysis or model.** |
| `nt_correction_source` | `known_nt`, `kc_override` or `predicted` |
| `super_class` | The shared levels in `config.SUPER_CLASS_VALUES`, including `sensory_ascending`, `efferent` and `vnc_intrinsic` |
| `super_class_source` | The source's raw value, so any mapping can be undone |

How `nt_known` is read: it joins evidence with `;` and `,`. Peptides, nitric oxide and `<x>-negative` entries are ignored. Tyramine and glycine count as transmitters, so "glutamate, tyramine" names two and is not used. R8 photoreceptors are listed with histamine and acetylcholine, which they co-release (Xia et al. 2023), so they keep the classifier's label. MaleCNS `nt_known`, where a repository filled it, is not used. Print `kenyon_cell_types(neurons)` once per dataset and confirm it holds only Kenyon cells.

`nt_known` is a per-neuron column and a few rows carry another type's entry (for example 28 Mi9 neurons, some with Mi4's "gaba, nitric oxide"). After the correction, some types still mix transmitters. If your analysis works per cell type, check the type's mix before using a single value.

The MaleCNS superclass mapping is `MALECNS_SUPER_CLASS`:

- `*_tbc` goes to its base class. The exception is `vnc_tbc` (1 neuron), which becomes `unknown` because "vnc" is not a class.
- `sensory_descending` becomes `sensory`, as FlyWire files labial-nerve sensory descending neurons.
- The efferent family (`cb_efferent`, `vnc_efferent`, `efferent_ascending`, `efferent_descending`) becomes `efferent`, a MaleCNS-only level. FlyWire has no such class: its brain efferent neurons are `motor` or `endocrine`.
- `ENS` (47 neurons) becomes `unknown`.
- A raw value not in the mapping raises.

A project that needs fewer classes merges levels in its own code. Merging `sensory_ascending` into `ascending` matches Schlegel et al. (2024): all 612 were `ascending` in annotation release v2.1.0.

## Reference numbers (FAFB, 783 proofread set)

Computed from annotation v3.2.0, with the ids of annotation v2.1.0 (139,255 rows, the published count) standing in for the proofread list. They match Project 4's counts on the real list. Your Step 0 numbers should match:

- Neuron set: 139,248 annotation rows; 7 dropped; 14 added; total 139,255.
- `super_class`: optic 77,537; central 32,383; sensory 16,904; visual_projection 8,038; ascending 1,750; descending 1,303; sensory_ascending 612; visual_centrifugal 524; motor 110; endocrine 80; unknown 14.
- `nt_correction_source = known_nt`: 77,169 rows, of which 18,513 change value: 5,173 Kenyon-cell rows, 9,799 to histamine (R1–6 and R7), 3,541 other changes (for example 1,603 GABA to glutamate). All 5,177 Kenyon cells become acetylcholine; dopamine falls from 5,909 neurons to 638.

MaleCNS numbers and the regions were not computed centrally. The hand-worked toys in the tests cover the rules; the Step 0 rerun checks the real numbers.

## The neuropil table

- **Keys** are FlyWire base names (`ME`, `MB_CA` …). Side markers (`_L`, `(R)`) are stripped automatically. Hemibrain-style names are mapped in `ALIASES`. Any name the code does not know raises, naming it; report it, do not add it privately.
- **`is_optic`** is True for the five optic neuropils (LA, ME, AME, LO, LOP) and for `Optic-unspecified`.
- **`region_group`** is drawn from anatomy, not from any result. The two unspecified rows have `region_group = unknown`.
- **No neuropil:** `unknown`, `UNASGD`, `None`, `<unspecified>` and `NotPrimary` give `region_group = unknown` and `is_optic = False`.

## Features, relative weight, randomiser

- `basic_features` returns `in_degree` and `out_degree` at the threshold you give, and `total_in_syn` and `total_out_syn` over all edges of the table you pass (cached at 1 synapse). For MaleCNS, pass `edges_b`.
- `relative_weight` divides each edge by the target's total input in the table you pass: the full FAFB edge table, or MaleCNS `edges_b`. Compute it before any filtering.
- The randomiser is unchanged: every in- and out-degree kept, no self-connections or duplicates, the same seed gives the same graph. Threshold first, then randomise. Report its warning if a dense graph cannot mix.
