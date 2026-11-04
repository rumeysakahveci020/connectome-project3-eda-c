# Step 0 on the corrected tables (eda_shared v0.2.0)

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
