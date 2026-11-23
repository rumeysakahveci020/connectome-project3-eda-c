# Week 9 extra controls, laptop-friendly. Resumable: if it stops, run it again. Run from the repo root in PowerShell.
# Step 1: learning rates for the two new conditions (seed 0, P=40, n=10, validation loss only; same grid as before)
python tools/week9_naive.py --tune --approved --conditions masked_free dense_units
# Step 2: small slice first (3 seeds, n=10, P=40 and 80): answers sign vs structure fastest
python tools/week9_naive.py --approved --conditions masked_free --seeds 3 --P 40 80 --n 10
python tools/week9_naive.py --approved --conditions dense_units --seeds 3 --P 40 80 --n 10
