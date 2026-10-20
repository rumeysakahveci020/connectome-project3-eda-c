"""Run a notebook cell by cell and print what it is doing, with times.

Same result as `jupyter nbconvert --execute --inplace`, but you can SEE progress, and the notebook
is saved after every cell, so a long run is never lost if something fails late.

    python tools/run_notebook.py notebooks/eda_C.ipynb
"""
import asyncio
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())   # silences the zmq warning

path = Path(sys.argv[1])
nb = nbformat.read(path, as_version=4)
code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
t0 = time.time()
state = {"start": t0}


def first_line(cell):
    for line in cell.source.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            return line.strip()[:80]
    return "(empty)"


def on_start(cell, cell_index, **_):
    if cell.cell_type != "code":
        return
    state["start"] = time.time()
    n = code_cells.index(cell_index) + 1 if cell_index in code_cells else "-"
    print(f"[{time.time() - t0:7.0f}s] cell {n}/{len(code_cells)} start : {first_line(cell)}", flush=True)


def on_done(cell, cell_index, execute_reply, **_):
    if cell.cell_type != "code":
        return
    print(f"[{time.time() - t0:7.0f}s]            done  ({time.time() - state['start']:.0f}s)", flush=True)
    nbformat.write(nb, path)                       # keep progress


client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
client.on_cell_start = on_start
client.on_cell_executed = on_done
try:
    client.execute()
except Exception as exc:                            # noqa: BLE001
    nbformat.write(nb, path)
    print(f"\nSTOPPED: {type(exc).__name__}: {str(exc)[-1500:]}", flush=True)
    sys.exit(1)
nbformat.write(nb, path)
print(f"\nfinished in {time.time() - t0:.0f}s", flush=True)
