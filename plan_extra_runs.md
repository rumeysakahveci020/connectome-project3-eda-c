# Extra runs plan (laptop-friendly)
Runs are resumable, so run in small batches overnight, one dataset at a time.

| Batch | What | Cells | Note |
|---|---|---|---|
| A | seeds 3-4, n in {1,3,10}, both datasets | 144 | same lr, same code |
| B | n = 30, P in {10..80}, 3 seeds | about 72 | P=40 n=30 took about 5 min |
| C | sign-free control (masked, no sign clipping) | 54 per dataset | separates sign from structure |
| D | equal-unit dense control | 54 per dataset | removes fifth term |
| E | n = 100 only at P=10,20 | few | P=320 n=100 > 1 h, skip |
Also: log rho = ||g_C||^2/||g||^2 and learning curves in each run.
