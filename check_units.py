# Run from the repo root (PowerShell: python check_units.py). Prints facts for the unit mismatch 5,670 vs 5,625.
import json, glob, subprocess, sys
print("Python", sys.version.split()[0])
for f in glob.glob("results/week9/*.json"): print(f)
rows=[json.loads(l) for l in open("results/week9/runs.jsonl")]
for ds in ("fafb","malecns"):
    r=[x for x in rows if x["dataset"]==ds and x["condition"]=="masked_sign"][0]
    print(ds,"units in run:",r["units"],"mask_weights:",r["mask_weights"])
print("Now find where the graph is built:")
subprocess.run(["git","grep","-n","-i","units\\|n_units\\|isolated\\|nodes","--","*.py"])
