# Project 3: Decomposing the Biological Prior — data cache and workstream C

Workstream C (neurotransmitters and local circuits) of the Week 3 EDA plan. Everything below can be regenerated from `data/cache`; the notebooks make no network calls.

## Layout
    config.py              Week 2 Data Conventions, unchanged
    eda_shared/            v0.1.1-eda, unchanged (not edited)
    src/                   fetch, cache, validators, C1-C6 code
    data/cache/            parquet caches (in .gitignore)
    notebooks/00_integration.ipynb   validators, assert_interchangeable, Step 0 counts
    notebooks/eda_C.ipynb            C1-C6
    figures/eda/           14 figures, named item_dataset_release_population[_threshold][_tag]
    reports/eda_C.md       one-page summary
    tools/                 run_notebook.py (progress), redo_figures_c2_c3.py, check_fafb_ids.py, synthetic dry run

## Pinned releases and sources
| | FlyWire FAFB | Janelia male CNS |
| --- | --- | --- |
| Release | materialisation 783 | v1.0, Traced neurons, synapse confidence >= 0.5 |
| Neurons | flywire_annotations tag v3.2.0 (Supplemental_file1_neuron_annotations.tsv) | neuPrint male-cns:v1.0 (properties and per-ROI counts) |
| Edges | Zenodo 10676866, proofread_connections_783.feather (Buhmann synapses, one row per pair and neuropil) | connectome-weights-male-cns-v1.0-minconf-0.5.feather (columns body_pre, body_post, weight) |
| Neuron list | proofread_root_ids_783.npy (139,255 ids) | status == Traced |

## Step 0 count check (run on our caches)
| Table | Expected | Ours |
| --- | --- | --- |
| fafb neurons | 139,248 (published 139,255) | **139,255** |
| fafb edges, syn_count >= 5 | 2,700,513 | 2,700,513 |
| malecns neurons | 165,122 | 165,122 |
| malecns edges, syn_count >= 5 | one agreed number | **6,235,649** (not yet agreed with Projects 1 and 4) |
| self-connections dropped | fafb 0, malecns 101 | fafb 0, malecns 101 |
| edges at 1 synapse | | fafb 15,091,983; malecns 25,563,096 |

Validators and `assert_interchangeable` pass on both datasets (00_integration.ipynb).

### FAFB neuron count
The annotation file has 139,248 ids, the proofread list 139,255. The two lists differ both ways: 14 proofread ids are missing from the annotations (all of them appear in the edges, 1,100 edges, 2,505 synapses; added with unknown annotations) and 7 annotation ids are not in the proofread list (no edges; dropped). Result: the 139,255 published neurons (`--fafb-proofread-ids`). Without that option the table has 139,262 rows.

## Source columns printed at the first run
FAFB annotations: cell_class, cell_sub_class, cell_type, dimorphism, fbbt_id, flow, fru_dsx, hartenstein_hemilineage, hemibrain_type, ito_lee_hemilineage, known_nt, known_nt_source, matching_notes, nerve, nucleus_id, pos_x/y/z, root_id, side, soma_x/y/z, status, super_class, supertype, supervoxel_id, synonyms, top_nt, top_nt_conf, vfb_id.
FAFB edges: ach_avg, da_avg, gaba_avg, glut_avg, neuropil, oct_avg, post_pt_root_id, pre_pt_root_id, ser_avg, syn_count.
Male CNS edges: body_post, body_pre, weight (no ROI, no confidence column; `edgeroi.neuropil` is therefore 'all' and the confidence filter is assumed from the file name).
Male CNS neuron properties used: bodyId, type, instance, class, subclass, superclass, somaSide, rootSide, predictedNt, predictedNtConfidence, consensusNt, pre, post, flywireType, mancType, hemibrainType, dimorphism, status, itoleeHl, roiInfo (region).

## Mappings
- side (FAFB): left, right, center; the source value `na` (30 neurons) -> unknown. Male CNS: somaSide, else rootSide; L, R, M -> left, right, center.
- nt_type: FAFB top_nt; male CNS **predictedNt** (Data Conventions D4), `unclear` -> unknown. consensusNt is kept in `data/cache/malecns_nt_consensus_v1.0.parquet` for the C2b check; it is not a column of the neurons table.
- region: the neuropil (FAFB) or primary ROI (male CNS) holding most of the neuron's synapses, counted as the Data Conventions do (one synapse per sender-receiver contact). **Group decision (15 Oct, D15): male CNS roiInfo `upstream + downstream`** (connection counts), the same unit as the FAFB edge table (FAFB input and output totals are equal, so FAFB counts outputs per partner too); `pre + post` counts release sites once and is not used. FAFB: the sum of `syn_count` over the neuron's edges as pre and as post. Ties go to the alphabetically first name (the tie rule is a week 3 decision; the counts of tied neurons are logged at build time).
- **super_class: provisional, not agreed.** FlyWire v3.2.0 has a tenth level `sensory_ascending` (612) that config.py lacks, and the male CNS has `cb_*`, `ol_*`, `vnc_*`, `ens`, `efferent_*` and `*_tbc` values with no FAFB counterpart. They are mapped in `src/provisional_maps.py` (guessed from the names; nerve-cord and unclear classes go to unknown).

## Deviations and open points
1. Three male CNS ROIs are not in `neuropils_provisional.csv`: `Optic-unspecified(L)`, `Optic-unspecified(R)`, `CentralBrain-unspecified`. Group decision: two rows, `Optic-unspecified` (is_optic True) and `CentralBrain-unspecified` (is_optic False), both region_group unknown, so these neurons are in the brain part (in_brain True), as the FAFB UNASGD neurons are kept in FAFB. The rows are passed with `add_provisional_flags(table=...)` (`GROUP_ROWS` in `src/eda_utils.py`); `eda_shared` itself is not edited, and a new `eda_shared` with the same rows would make the addition a no-op. Evidence: the second ROI of the Optic-unspecified neurons is ME/LA/LO/LOP/AME for 98-100% of them, that of the CentralBrain-unspecified neurons is never optic, and the named ROI holds a median of only 24-37% of their synapses (other students' counts). FAFB UNASGD neurons keep is_optic False and region_group unknown from `eda_shared`; in_brain is False for them, but FAFB is not filtered by in_brain, so they stay in the data. Unspecified neuron counts depend on the source (761, 750, 745 in other students' tables); this cache's counts are in `reports/eda_C_numbers.json`.
2. Male CNS edge count at 5 synapses: 6,235,649; to be agreed with Projects 1 and 4.
3. Male CNS `predictedNt` labels 99.9% of Kenyon-type neurons dopamine; `consensusNt` says acetylcholine (C2b).
4. C6 uses a symmetric log axis for the z-scores, because a single motif (111U, z about -1,600) hides the others on a linear axis. The triad census is computed with igraph (the plan names networkx triadic_census); the counts are identical, checked by a test.
5. The male CNS `ens` super_class value and the `*_tbc` suffix were not checked against the male CNS documentation.
6. FAFB `UNASGD` neurons (no neuropil holds their synapses): `eda_shared` gives them is_optic False and region_group unknown, so no new row is needed for FAFB; in_brain is False for them, but FAFB is not filtered by in_brain, so they stay in the data. Another student's list of them had 6 neurons (3 ascending, JO-EV6, JO-unclear and one R1-6 photoreceptor); the photoreceptor is counted as central too, because one neuron does not need its own rule. This cache's list: `python tools/check_unassigned.py`.
7. **Open with the group and the instructor (as of 15 Oct):** a new `eda_shared` version that contains the two rows (until then they are passed with `table=`; the new version makes that a no-op and the results stay the same); the tie rule of D15 (ties go to the alphabetically first name here: 319 FAFB and 81 male CNS neurons (after the group region rule); another student saw tied neurons change order when the label spelling changed); and a shared nerve-cord ROI list (this repo uses its own `NERVE_CORD_ROIS` in `src/eda_utils.py`).

## Rebuild everything from scratch and verify (recommended)
    python tools/rebuild_all.py --dry-run     # shows the plan
    python tools/rebuild_all.py               # preflight, backup, tests, both caches, both notebooks, summary, verification
    python tools/verify_outputs.py            # on its own: recomputes every number from the cache, tests the C2 colours,
                                              # checks stale figures, notebook errors, the report and the zip (no .env, no data)
                                              # D2 recomputes C1, C3, C4, C5, C6 with different code (numpy, scipy, loops), so a bug inside a project function cannot pass

## How to run
    pip install -r requirements.txt
    python -m pytest eda_shared/tests tests
    python -m src.build_cache --fafb-edges data/raw/proofread_connections_783.feather \
        --fafb-annotations data/raw/Supplemental_file1_neuron_annotations.tsv \
        --fafb-proofread-ids data/raw/proofread_root_ids_783.npy \
        --malecns-edges data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather --provisional-maps
    $env:EDA_UNMAPPED_REGION="unknown"; $env:EDA_WORKERS="4"; $env:EDA_N_SEEDS="100"
    python tools/run_notebook.py notebooks/eda_C.ipynb       # figures and report
Male CNS neurons come from neuPrint: put `NEUPRINT_TOKEN=...` in a gitignored `.env`.

## Agreed numbers (fill in after the session)
8. This cache (rebuilt with the group rule): the unspecified ROIs hold 761 neurons (365 CentralBrain-unspecified, 359 Optic-unspecified(L), 37 Optic-unspecified(R)), the same as another student's count. FAFB UNASGD neurons: 9 (7 ascending AN, 1 JO-EV6, 1 R7 photoreceptor); the other 616 of the 625 FAFB neurons in region_group unknown have no edge at all.
9. The group region rule changes the male CNS mushroom body selection (by region, tau = 10): 4,468 neurons, 34,963 edges, 12 isolated under pre + post; 4,435, 30,314, 48 under upstream + downstream. `python tools/compare_region_rules.py` lists the neurons that moved.
