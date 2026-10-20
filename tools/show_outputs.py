"""Print the text and table outputs saved inside an executed notebook (no re-run).

    python tools/show_outputs.py notebooks/eda_C.ipynb
Shows each code cell's printed text and tables; figures are skipped (they are in figures/eda).
"""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
k = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    k += 1
    outs = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            outs.append("".join(o["text"]).rstrip())
        elif "data" in o and "text/plain" in o["data"]:
            outs.append("".join(o["data"]["text/plain"]).rstrip())
        elif o.get("output_type") == "error":
            outs.append("ERROR: " + o.get("ename", "") + ": " + o.get("evalue", ""))
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    first = next((ln.strip() for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")), "")
    print(f"\n===== cell {k}: {first[:70]}")
    for t in outs:
        if t.strip() and not t.strip().startswith("<Figure"):
            print(t)
