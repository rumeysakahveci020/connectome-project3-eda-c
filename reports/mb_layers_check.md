# Are the input and output layers complete? (region rule against MB cell types)

## FAFB

| | all neurons of the type | kept by the region rule | in the MB cell-type selection | region rule or cell type (union) |
| --- | --- | --- | --- | --- |
| Kenyon cells (type starts with KC) | 5,177 | 4,926 | 5,177 | 5,177 |
| MBONs (type starts with MBON) | 96 | 22 | 96 | 96 |
| all units | | 5,258 | 5,612 | 5,670 |

MB-type neurons the region rule drops: 412. Where they went (their region under the shared rule):

| kind | region | neurons |
| --- | --- | --- |
| KC | CRE | 158 |
| KC | PLP | 84 |
| other MB type | CRE | 65 |
| MBON | CRE | 27 |
| MBON | SMP | 22 |
| other MB type | SMP | 12 |
| MBON | SIP | 9 |
| KC | SCL | 7 |
| MBON | LH | 7 |
| MBON | SLP | 4 |
| MBON | LAL | 4 |
| other MB type | SLP | 4 |
| other MB type | LH | 2 |
| other MB type | ATL | 2 |
| KC | SMP | 1 |
| MBON | SCL | 1 |
| KC | SIP | 1 |
| other MB type | FB | 1 |

MBON types: how many neurons per type the region rule keeps.

| MBON type | neurons | kept |
| --- | --- | --- |
| MBON10 | 9 | 0 |
| MBON25,MBON34 | 6 | 0 |
| MBON15-like | 5 | 0 |
| MBON19 | 4 | 0 |
| MBON15 | 4 | 0 |
| MBON12 | 4 | 0 |
| MBON09 | 4 | 4 |
| MBON14 | 4 | 4 |
| MBON07 | 4 | 2 |
| MBON01 | 2 | 1 |
| MBON02 | 2 | 0 |
| MBON03 | 2 | 1 |
| MBON04 | 2 | 0 |
| MBON11 | 2 | 2 |
| MBON13 | 2 | 0 |
| MBON16 | 2 | 1 |
| MBON17 | 2 | 0 |
| MBON05 | 2 | 1 |
| MBON06 | 2 | 2 |
| MBON18 | 2 | 1 |
| MBON17-like | 2 | 0 |
| MBON21 | 2 | 0 |
| MBON20 | 2 | 0 |
| MBON22 | 2 | 2 |
| MBON23 | 2 | 0 |
| MBON24 | 2 | 1 |
| MBON26 | 2 | 0 |
| MBON27 | 2 | 0 |
| MBON28 | 2 | 0 |
| MBON29 | 2 | 0 |
| MBON30 | 2 | 0 |
| MBON31 | 2 | 0 |
| MBON32 | 2 | 0 |
| MBON33 | 2 | 0 |
| MBON35 | 2 | 0 |

Median share of input sites (post) and output sites (pre) inside mushroom body neuropils:

| kind | kept by the region rule | neurons | inputs in MB | outputs in MB |
| --- | --- | --- | --- | --- |
| KC | False | 251 | 0.31 | 0.51 |
| KC | True | 4,926 | 0.89 | 0.82 |
| MBON | False | 74 | 0.4 | 0.02 |
| MBON | True | 22 | 0.8 | 0.11 |

## MaleCNS

| | all neurons of the type | kept by the region rule | in the MB cell-type selection | region rule or cell type (union) |
| --- | --- | --- | --- | --- |
| Kenyon cells (type starts with KC) | 4,064 | 4,064 | 4,064 | 4,064 |
| MBONs (type starts with MBON) | 97 | 31 | 97 | 97 |
| all units | | 4,435 | 4,510 | 4,515 |

MB-type neurons the region rule drops: 80. Where they went (their region under the shared rule):

| kind | region | neurons |
| --- | --- | --- |
| MBON | CRE | 26 |
| MBON | SLP | 15 |
| MBON | SMP | 11 |
| MBON | LAL | 7 |
| other MB type | CRE | 5 |
| other MB type | SLP | 4 |
| MBON | PLP | 2 |
| other MB type | LH | 2 |
| other MB type | ATL | 2 |
| MBON | CAN | 1 |
| MBON | CentralBrain-unspecified | 1 |
| MBON | SCL | 1 |
| MBON | LH | 1 |
| MBON | SIP | 1 |
| other MB type | SMP | 1 |

MBON types: how many neurons per type the region rule keeps.

| MBON type | neurons | kept |
| --- | --- | --- |
| MBON10 | 9 | 0 |
| MBON12 | 4 | 0 |
| MBON09 | 4 | 4 |
| MBON07 | 4 | 1 |
| MBON14 | 4 | 4 |
| MBON25-like | 4 | 0 |
| MBON19 | 4 | 0 |
| MBON15-like | 4 | 0 |
| MBON15 | 4 | 0 |
| MBON01 | 2 | 0 |
| MBON04 | 2 | 0 |
| MBON03 | 2 | 2 |
| MBON02 | 2 | 0 |
| MBON05 | 2 | 2 |
| MBON13 | 2 | 2 |
| MBON06 | 2 | 2 |
| MBON11 | 2 | 2 |
| MBON17 | 2 | 2 |
| MBON17-like | 2 | 0 |
| MBON18 | 2 | 2 |
| MBON16 | 2 | 2 |
| MBON20 | 2 | 0 |
| MBON21 | 2 | 2 |
| MBON23 | 2 | 0 |
| MBON22 | 2 | 0 |
| MBON24 | 2 | 0 |
| MBON25 | 2 | 1 |
| MBON26 | 2 | 0 |
| MBON27 | 2 | 0 |
| MBON28 | 2 | 0 |
| MBON29 | 2 | 1 |
| MBON30 | 2 | 2 |
| MBON31 | 2 | 0 |
| MBON32 | 2 | 0 |
| MBON33 | 2 | 0 |
| MBON34 | 2 | 0 |
| MBON35 | 2 | 0 |

Median share of input sites (post) and output sites (pre) inside mushroom body neuropils:

| kind | kept by the region rule | neurons | inputs in MB | outputs in MB |
| --- | --- | --- | --- | --- |
| KC | True | 4,064 | 0.93 | 0.95 |
| MBON | False | 66 | 0.52 | 0.05 |
| MBON | True | 31 | 0.89 | 0.23 |

