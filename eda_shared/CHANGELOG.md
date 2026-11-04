# Changelog

## v0.2.0 — Week 4 Shared Data Corrections

Replaces v0.1.1-eda. Ships with `config.py` v0.2.0. Step 0 numbers change: rerun it.

**New: `corrections.py`** (item numbers refer to *Week 4 — Shared Data Corrections*)
- `add_nt_type_corrected` builds `nt_type_corrected` and `nt_correction_source` (1). Also `parse_known_nt`, `single_known_nt`, `is_kenyon_cell`, `kenyon_cell_types`, `nt_correction_summary`.
- `map_super_class` maps the raw FAFB `super_class` or MaleCNS `superclass` to the shared levels and keeps the raw value in `super_class_source` (2).
- `align_to_proofread` builds the FAFB neuron set: the 783 proofread ids, 139,255 neurons (3).
- `malecns_brain_part` crops MaleCNS by synapse location and returns the brain-part neurons, edges, edgeroi and the Step 0 counts (4).
- `region_from_counts` applies one region rule to both datasets, from all synapses; `fafb_neuropil_counts` and `malecns_neuropil_counts` build its input (6). Also `neuropil_location`.

**`eda_helpers.py`**
- New constant `MALECNS_NERVE_CORD` (4).
- `NO_NEUROPIL` also holds `None`, `<unspecified>` and `NotPrimary`, the spellings for a synapse outside every neuropil.
- `relative_weight`: docstring corrected; for MaleCNS pass the brain-part edges (5). The computation is unchanged.
- `add_provisional_flags`: docstring says which `nerve_cord` to pass; one pandas FutureWarning removed. Results unchanged.

**`neuropils_provisional.csv`**
- Two rows added: `Optic-unspecified` (`is_optic` True) and `CentralBrain-unspecified` (`is_optic` False), both `region_group = unknown` (6). The file name is kept so existing imports work.

**`config.py` v0.2.0** (outside this folder; replace yours)
- `CONFIG_VERSION = "0.2.0"`.
- `SUPER_CLASS_VALUES` gains `sensory_ascending`, `efferent`, `vnc_intrinsic`; `NT_CORRECTION_SOURCES` added.
- `NEURON_SCHEMA` gains `nt_type_corrected`, `nt_correction_source`, `super_class_source`. Validators that check the schema will ask for these columns until the tables are rebuilt; that is intended.

**Tests:** 49 (the 15 from v0.1.1, two of them updated for the new table rows, and 34 new). They pass with pandas 2.3.3 / numpy 2.2.6 and with pandas 3.0.

**Unchanged:** `randomiser.py`, `basic_features`, `canonical_neuropil`, every v0.1.1 function signature.

**What each repository deletes after copying**
- Local copies of the nerve-cord list and of the two unspecified rows (for example `src/provisional.py` in Project 4, `MALECNS_NERVE_CORD` and `PROPOSED_NEUROPIL_ROWS` in Project 5's `src/eda_e.py`).
- Local region functions used for the shared table (a different tie rule stays allowed in project code, stated as such).
- Local superclass maps for the shared table (merging levels on top of `super_class` stays allowed).
