# Numbers for the Project 3 lock note (corrected tables)

## FAFB: 5,670 mushroom body neurons (region); cell-type selection 5,612; overlap 5,200; is_optic True: 0
- labels (nt_type_corrected): {'acetylcholine': 5273, 'dopamine': 333, 'glutamate': 33, 'gaba': 29, 'unknown': 1, 'octopamine': 1}; source: {'known_nt': 5611, 'predicted': 59}; unknown 0.0%; KC 5177, MBON 96
- signs, units: rule 1 {'plus': 5273, 'minus': 62, 'free': 335}, rule 2 {'plus': 5306, 'minus': 29, 'free': 335}
- glutamate units whose sign changes between the two rules: 33
- type mix: {'types': 25, 'mixed_types': 0, 'neurons_in_mixed_types': 0}; ties: {'neurons_with_a_tie': 232, 'tied_and_selected_now': 41, 'tied_selected_but_could_leave': 0, 'tied_not_selected_but_could_join': 0, 'selected_total': 5670}

| tau | edges | per neuron | isolated | no input | no output | median in/out | fast-signed source | KC to MBON kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 536,343 | 94.59 | 2 | 4 | 3 | 73/88 | 90% | 100% (62,261) |
| 3 | 91,421 | 16.12 | 17 | 29 | 35 | 5/13 | 94% | 58% (35,943) |
| 5 | 47,329 | 8.35 | 45 | 124 | 68 | 2/6 | 96% | 34% (21,438) |
| 10 | 21,961 | 3.87 | 151 | 454 | 190 | 1/3 | 97% | 7% (4,593) |

## MaleCNS: 4,515 mushroom body neurons (region); cell-type selection 4,510; overlap 4,430; is_optic True: 0
- labels (nt_type_corrected): {'acetylcholine': 4115, 'dopamine': 332, 'gaba': 29, 'glutamate': 25, 'unknown': 12, 'serotonin': 2}; source: {'kc_override': 4064, 'predicted': 451}; unknown 0.3%; KC 4064, MBON 97
- signs, units: rule 1 {'plus': 4115, 'minus': 54, 'free': 346}, rule 2 {'plus': 4140, 'minus': 29, 'free': 346}
- glutamate units whose sign changes between the two rules: 25
- type mix: {'types': 25, 'mixed_types': 1, 'neurons_in_mixed_types': 12}; ties: {'neurons_with_a_tie': 76, 'tied_and_selected_now': 4, 'tied_selected_but_could_leave': 0, 'tied_not_selected_but_could_join': 0, 'selected_total': 4515}

| tau | edges | per neuron | isolated | no input | no output | median in/out | fast-signed source | KC to MBON kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,027,369 | 227.55 | 0 | 0 | 0 | 185/201 | 86% | 100% (61,210) |
| 3 | 241,340 | 53.45 | 0 | 1 | 1 | 35/46 | 88% | 72% (44,042) |
| 5 | 103,580 | 22.94 | 1 | 11 | 6 | 9/18 | 89% | 55% (33,496) |
| 10 | 36,138 | 8.00 | 12 | 151 | 21 | 2/7 | 91% | 27% (16,779) |

C4 on the corrected labels: {'neuropils': 12, 'flip': 7, 'flip_names': ['FB', 'GNG', 'IPS', 'PLP', 'PVLP', 'SMP', 'SPS'], 'shifted_025': 9, 'shift_hi': 1.1532467532467532}
Cross-dataset: {'median_syn_per_edge_tau1': {'fafb': 1.0, 'malecns': 1.0}, 'edges_per_neuron': {'1': {'fafb': 94.59312169312169, 'malecns': 227.54573643410853}, '3': {'fafb': 16.12363315696649, 'malecns': 53.452934662236984}, '5': {'fafb': 8.34726631393298, 'malecns': 22.941306755260243}, '10': {'fafb': 3.8731922398589065, 'malecns': 8.003986710963455}}, 'tau_malecns_matching_fafb_tau5_density': 10, 'fafb_edges_per_neuron_tau5': 8.34726631393298}
