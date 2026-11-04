# Numbers for the Project 3 lock note (corrected tables)

## FAFB: 5,258 mushroom body neurons (region); cell-type selection 5,612; overlap 5,200; is_optic True: 0
- labels (nt_type_corrected): {'acetylcholine': 4980, 'dopamine': 246, 'glutamate': 16, 'gaba': 14, 'unknown': 1, 'octopamine': 1}; source: {'known_nt': 5228, 'predicted': 30}; unknown 0.0%; KC 4926, MBON 22
- signs, units: rule 1 {'plus': 4980, 'minus': 30, 'free': 248}, rule 2 {'plus': 4996, 'minus': 14, 'free': 248}
- type mix: {'types': 22, 'mixed_types': 0, 'neurons_in_mixed_types': 0}; ties: {'neurons_with_a_tie': 232, 'tied_and_selected_now': 35, 'tied_selected_but_could_leave': 2, 'tied_not_selected_but_could_join': 5, 'selected_total': 5258}

| tau | edges | per neuron | isolated | no input | no output | median in/out | fast-signed source | KC to MBON kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 448,478 | 85.29 | 4 | 7 | 5 | 69/77 | 90% | 100% (25,958) |
| 3 | 70,427 | 13.39 | 17 | 31 | 39 | 5/10 | 94% | 77% (19,978) |
| 5 | 36,570 | 6.96 | 55 | 114 | 85 | 2/5 | 97% | 51% (13,210) |
| 10 | 18,653 | 3.55 | 145 | 353 | 165 | 1/2 | 99% | 10% (2,696) |

## MaleCNS: 4,435 mushroom body neurons (region); cell-type selection 4,510; overlap 4,430; is_optic True: 0
- labels (nt_type_corrected): {'acetylcholine': 4080, 'dopamine': 323, 'gaba': 14, 'glutamate': 9, 'unknown': 9}; source: {'kc_override': 4064, 'predicted': 371}; unknown 0.2%; KC 4064, MBON 31
- signs, units: rule 1 {'plus': 4080, 'minus': 23, 'free': 332}, rule 2 {'plus': 4089, 'minus': 14, 'free': 332}
- type mix: {'types': 24, 'mixed_types': 1, 'neurons_in_mixed_types': 12}; ties: {'neurons_with_a_tie': 76, 'tied_and_selected_now': 4, 'tied_selected_but_could_leave': 0, 'tied_not_selected_but_could_join': 0, 'selected_total': 4435}

| tau | edges | per neuron | isolated | no input | no output | median in/out | fast-signed source | KC to MBON kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 981,780 | 221.37 | 0 | 0 | 0 | 184/193 | 86% | 100% (26,557) |
| 3 | 216,207 | 48.75 | 0 | 1 | 1 | 35/41 | 87% | 83% (22,090) |
| 5 | 87,431 | 19.71 | 6 | 14 | 18 | 9/15 | 88% | 72% (19,194) |
| 10 | 30,314 | 6.84 | 48 | 167 | 73 | 2/6 | 91% | 45% (11,974) |

C4 on the corrected labels: {'neuropils': 12, 'flip': 7, 'flip_names': ['FB', 'GNG', 'IPS', 'PLP', 'PVLP', 'SMP', 'SPS'], 'shifted_025': 9, 'shift_hi': 1.1532467532467532}
Cross-dataset: {'median_syn_per_edge_tau1': {'fafb': 1.0, 'malecns': 1.0}, 'edges_per_neuron': {'1': {'fafb': 85.29440852034995, 'malecns': 221.37091319052988}, '3': {'fafb': 13.394256371243818, 'malecns': 48.75016910935739}, '5': {'fafb': 6.955116013693419, 'malecns': 19.713866967305524}, '10': {'fafb': 3.5475465956637504, 'malecns': 6.835174746335964}}, 'tau_malecns_matching_fafb_tau5_density': 10, 'fafb_edges_per_neuron_tau5': 6.955116013693419}
