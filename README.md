# Project 3: Decomposing the Biological Prior: data cache and workstream C

Workstream C (neurotransmitters and local circuits) of the EDA, rebuilt on the **corrected shared tables** (`eda_shared` v0.2.0, Week 4 *Shared Data Corrections*). Everything below can be regenerated from `data/cache`; the notebooks make no network calls.

## Layout
    config.py              v0.2.0, copied unchanged from the shared package
    eda_shared/            v0.2.0, copied unchanged (not edited); pytest eda_shared/tests -> 49 tests
    src/                   fetch, cache, validators, C1-C6 code
    data/raw/              downloads (in .gitignore)      data/cache/   parquet caches (in .gitignore)
    notebooks/00_integration.ipynb   validators, assert_interchangeable, Step 0 counts
    notebooks/eda_C.ipynb            C1-C6, C5b (mushroom body type mix)
    figures/eda/           14 figures drawn on the corrected tables
    figures/eda_v0.1.1/    the 14 figures from before the corrections, kept so the change is visible
    reports/eda_C.md       one-page summary         reports/step0_C.md   Step 0 on the corrected tables
    tools/                 rebuild_all.py, step0.py, verify_outputs.py, final_check.py, make_summary.py, synthetic dry run

## What changed with the corrections (and what this repository does about it)
| # | Correction | Where it is applied |
| --- | --- | --- |
| 1 | `nt_type_corrected` (FAFB: `nt_known` where it names one transmitter; male CNS: Kenyon-cell types are acetylcholine) | `add_nt_type_corrected` in `src/fetch_fafb.py` and `src/fetch_malecns.py`. **C1, C3, C4 and C5b use `nt_type_corrected`.** C2 studies the classifier's own output, so it keeps `nt_type` and `nt_confidence`. |
| 2 | Shared `super_class` levels, raw value kept in `super_class_source` | `map_super_class`; the local provisional map (`src/provisional_maps.py`) is deleted |
| 3 | FAFB neuron set = the 783 proofread list (139,255) | `align_to_proofread` |
| 4 | Male CNS brain part cropped by synapse location | `malecns_brain_part`; needs edgeroi with one row per primary neuropil (below) |
| 5 | Relative-weight denominator = brain-part input | not used in workstream C (no relative weights); noted for Project 3's later work |
| 6 | One region rule for both datasets (all synapses, inputs plus outputs, ties alphabetical) | `region_from_counts` with `fafb_neuropil_counts` and `malecns_neuropil_counts`; the local region functions and the two group rows are deleted |

## Inputs (`data/raw/`)
| Dataset | File | Used for |
| --- | --- | --- |
| FAFB | `proofread_connections_783.feather` (Zenodo 10676866, Buhmann synapses, one row per pair and neuropil) | edges, edgeroi |
| FAFB | `Supplemental_file1_neuron_annotations.tsv`, annotation release v3.2.0 | labels |
| FAFB | `proofread_root_ids_783.npy` (Zenodo 10676866) | the neuron set |
| FAFB | **new:** `per_neuron_neuropil_count_pre_783.feather`, `per_neuron_neuropil_count_post_783.feather` (Zenodo 10676866) | region |
| male CNS | neuPrint `male-cns:v1.0` (`NEUPRINT_TOKEN` in a gitignored `.env`): neuron table, per-ROI table, primary ROIs; saved as `neuprint_malecns_v1.0_traced.pkl` | labels, region |
| male CNS | **new:** `syn-partners-male-cns-v1.0-minconf-0.5.feather` (6.8 GB, one row per synapse with `primary_post`) | edgeroi with one row per primary neuropil |
| male CNS | optional: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | cross-check of the per-neuropil counts |

**How the male CNS edgeroi is built.** `read_syn_partners` streams the syn-partners file batch by batch and counts synapses per (pre, post, `primary_post`) between Traced neurons; a missing `primary_post` is `<unspecified>`, self-connections are dropped (D9). The sum over neuropils is checked against `connectome-weights` (`weights_check` in `data/cache/step0_malecns.json`). Which file defines the per-neuropil table was not stated in the Corrections; this construction should be confirmed with the group.

## Build, Step 0, rebuild
    python -m pytest eda_shared/tests tests            # 49 shared tests + ours
    python -m src.build_cache --fafb-edges data/raw/proofread_connections_783.feather \
        --fafb-annotations data/raw/Supplemental_file1_neuron_annotations.tsv --fafb-proofread-ids data/raw/proofread_root_ids_783.npy \
        --fafb-count-pre data/raw/per_neuron_neuropil_count_pre_783.feather --fafb-count-post data/raw/per_neuron_neuropil_count_post_783.feather \
        --malecns-syn-partners data/raw/syn-partners-male-cns-v1.0-minconf-0.5.feather \
        --malecns-weights data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather
    python tools/step0.py                              # the Step 0 table (also reports/step0_C.md); paste it below
    python tools/rebuild_all.py --skip-build           # tests, notebooks (100 null graphs), summary, verification (about 25 minutes)

## Step 0 on the corrected tables
(paste `reports/step0_C.md` here after the build; the first repository to finish posts its table to the group and every other one must match it exactly)

## Mappings
- side (FAFB): left, right, center; the source value `na` -> unknown. Male CNS: somaSide, else rootSide; L, R, M -> left, right, center.
- nt_type: FAFB `top_nt`; male CNS `predictedNt` (Data Conventions D4), `unclear` -> unknown. `nt_type_corrected` = the Week 4 rule. consensusNt is kept in `data/cache/malecns_nt_consensus_v1.0.parquet` for the C2b check; it is not a column of the neurons table.
- region: Week 4 correction 6 (see the table above).

## Deviations and open points
1. C6 uses a symmetric log axis for the z-scores, because a single motif (111U) hides the others on a linear axis. The triad census is computed with igraph (the plan names networkx `triadic_census`); the counts are identical, checked by a test.
2. Male CNS edge count at 5 synapses: see `reports/step0_C.md`; to be agreed with the group.
3. The male CNS `nt_type_corrected` is the Kenyon-cell override only; `consensusNt` differs elsewhere (agreement in C2b). A project that wants `consensusNt` for all male CNS neurons builds its own column.
4. The mushroom-body selection is by region (neuropils MB_CA, MB_PED, MB_VL, MB_ML); the week 2 subgraph was selected by cell type, so both are shown in C5.
5. Files removed because the shared package replaced them: `src/provisional_maps.py`, `tools/check_roiinfo_rule.py`, `tools/compare_region_rules.py`, `tools/check_unspecified_regions.py`, `tools/check_unassigned.py`, `tools/check_fafb_ids.py`, `tools/redo_figures_c1.py`, `tools/redo_figures_c2_c3.py`.

## Dry run without the real downloads
    python tools/make_synthetic_raw.py data/synthetic          # prints the build command for the synthetic files
Nothing in the synthetic files is real data; never put those numbers in a report or a figure you hand in.
