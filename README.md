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
| Quantity | Value | Note |
| --- | --- | --- |
| FAFB neurons | 139,255  OK | the 783 proofread list |
| FAFB annotation rows kept | 139,241  OK |  |
| FAFB proofread ids added (unknown labels) | 14  OK |  |
| FAFB annotation rows dropped | 7  OK |  |
| FAFB super_class optic | 77,537  OK |  |
| FAFB super_class central | 32,383  OK |  |
| FAFB super_class sensory | 16,904  OK |  |
| FAFB super_class visual_projection | 8,038  OK |  |
| FAFB super_class ascending | 1,750  OK |  |
| FAFB super_class descending | 1,303  OK |  |
| FAFB super_class sensory_ascending | 612  OK |  |
| FAFB super_class visual_centrifugal | 524  OK |  |
| FAFB super_class motor | 110  OK |  |
| FAFB super_class endocrine | 80  OK |  |
| FAFB super_class unknown | 14  OK |  |
| FAFB edges at tau = 1 | 15,091,983 |  |
| FAFB edges at >= 5 synapses | 2,700,513  OK | published Buhmann count |
| FAFB neurons with region unknown | 281 |  |
| FAFB region ties (alphabetical rule) | 232 |  |
| FAFB regions (neurons per region, top 8) | ME_R 29,012; ME_L 28,691; LO_L 8,027; LOP_R 7,033; LO_R 6,071; GNG 5,954; LOP_L 5,436; LA_L 5,083 |  |
| FAFB self-connections dropped | 0  OK |  |
| MaleCNS Traced neurons (before the crop) | 165,122  OK |  |
| MaleCNS brain part: neurons kept | 144,414 |  |
| MaleCNS brain part: neurons dropped | 20,708 |  |
| MaleCNS ascending kept | 1,841 |  |
| MaleCNS sensory_ascending kept | 534 |  |
| MaleCNS brain edges at tau = 1 | 21,706,010 |  |
| MaleCNS brain edges at >= 5 synapses | 5,143,528 |  |
| MaleCNS synapses in brain / nerve cord / no location | 99,982,458 / 24,041,864 / 251 |  |
| MaleCNS self-connections dropped | 101  OK |  |
| MaleCNS per-neuropil counts equal connectome-weights | True | {'weights_pairs': 25563096, 'weights_synapses': 124024573, 'syn_partners_pairs': 25563096, 'syn_partners_synapses': 124024573, 'agree': True} |
| FAFB central population (is_optic False, region known) | 45,203 |  |
| MaleCNS central population (is_optic False, region known) | 45,642 |  |
| Transmitter correction fafb / known_nt | 77,169 rows, 18,513 change |  |
| Transmitter correction fafb / predicted | 62,086 rows, 0 change |  |
| Transmitter correction malecns / kc_override | 4,064 rows, 4,064 change |  |
| Transmitter correction malecns / predicted | 140,350 rows, 0 change |  |
| FAFB dopamine neurons before -> after | 5,909 -> 638 | reference 5,909 -> 638 |
| Kenyon-cell types (print once, confirm only Kenyon cells) | fafb:KCa'b'-ap1 1; fafb:KCab 1,643; fafb:KCab-p 128; fafb:KCapbp-ap1 280; fafb:KCapbp-ap2 298; fafb:KCapbp-m 338; fafb:KCg-d 295; fafb:KCg-m 2,189; fafb:KCg-s1 2; fafb:KCg-s2 2; fafb:KCg-s3 1; malecns:KC 2; malecns:KCa'b'-ap1 199; malecns:KCa'b'-ap2 291; malecns:KCa'b'-m 205; malecns:KCab-c 488; malecns:KCab-m 536; malecns:KCab-p 129; malecns:KCab-s 657; malecns:KCg 1; malecns:KCg-d 206; malecns:KCg-m 1,342; malecns:KCg-s1 2; malecns:KCg-s2 2; malecns:KCg-s3 2; malecns:KCg-s4 2 |  |


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


---

## Week 9: naive result and failure argument

Decomposing the Biological Prior: FlyWire FAFB 783 and MaleCNS v1.0 mushroom-body subgraph.
Tag `week9`. Results: `results/week9/`, figures: `figures/week9/`, notebook: `notebooks/week9_naive.ipynb`.

## Locked choices
| Item | Value |
|---|---|
| Graph | Union selection (region rule OR mushroom body cell type), tau = 5 |
| FlyWire | 5,670 units, 47,329 edges (KC 5,177, MBON 96) |
| Male CNS | 4,515 units, 103,580 edges (KC 4,064, MBON 97) |
| Task | Random binary patterns, d = 50, f = 0.5, noise 0.1; validation and test on fresh noisy copies |
| P | 10, 20, 40, 80, 160, 320 |
| n (copies per pattern) | 1, 3, 10 finished (30, 100 planned) |
| Model | Recurrent ReLU, T = 3 steps (longest KC to MBON path + 1) |
| Conditions | `masked_sign` (real mask and signs, clipped to sign after every Adam step); `dense_params` (parameter-matched dense, H = 218 FlyWire / 322 male CNS) |
| Training | Adam, 200 epochs, early stopping patience 20, 3 seeds |
| Learning rate | Chosen once per (dataset, condition) on seed 0 at P = 40, n = 10: masked 0.01 (upper edge of grid), dense 0.003 |
| Metric | Test accuracy; majority baseline 0.51-0.57 |

## Deviations from the plan
- T = 2 in early code never read direct KC to MBON edges; fixed to T = 3.
- n = 30 and 100, seeds 3-4 and 10 seeds not run (laptop time limit; one cell at P = 320, n = 100 takes over 1 hour).
- Sign-free and equal-unit dense controls not run yet.
- Masked lr is at the grid edge, so it may be under-tuned.
- Run graphs have 5,625 (FlyWire) and 4,514 (male CNS) units versus 5,670 and 4,515 in the lock note. Edges are identical. The code removes units with no connection (build_units, isolated_removed); the exact counts (45 and 1) are to be confirmed with check_units.py.
- One extra P = 40, n = 30 row in `runs.jsonl` is a timing run, excluded from tables.

## Main findings (3-seed means)
- n = 10: no clear winner; differences are 0.00-0.05, close to seed spread (0.001-0.04).
- n = 1: masked higher (0.77 vs 0.66 FlyWire, 0.76 vs 0.67 male CNS), but dense runs collapse to chance: 7/54 FlyWire, 5/54 male CNS, masked 0.
- Clipping: 13.0% (FlyWire) and 15.4% (male CNS) of weights clipped per step; 14-15% end exactly at zero.
- These are preliminary observations, not a final comparison. One contrast cannot separate sign, structure, degree and sparsity.

## Reproduce
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/week9_naive.ipynb
```

## To do
Seeds 5-9 and seeds 3-4 outside the key cells, n = 30/100, sign-free control on the full grid, equal-unit dense control, measure rho and per-step learning curves, confirm the unit-count difference with `check_units.py`, pinned environment file, update report and slides.
