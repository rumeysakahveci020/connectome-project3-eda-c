"""Replace the embedded C2 and C3 images of the executed notebook with the current files in figures/eda.

Use it after `python tools/redo_figures_c2_c3.py`. Tables, text and every other output stay as they are.
No re-run, no new numbers: the images come from your own cache through the figure files.

    python tools/refresh_notebook_images.py
"""
import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import eda_utils as U  # noqa: E402

path = ROOT / "notebooks" / "eda_C.ipynb"
nb = json.load(open(path, encoding="utf-8"))
done = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if 'U.fig_name("C1"' in src:                       # eight images in one cell, in loop order
        names = [U.fig_name("C1", "both", pop, extra=by + ("_weighted" if w else ""))
                 for pop in ("central", "optic") for by in ("super_class", "region_group") for w in (False, True)]
        imgs = [o for o in cell.get("outputs", []) if "data" in o and "image/png" in o["data"]]
        if len(imgs) != len(names) or not all((U.FIG_DIR / n).exists() for n in names):
            print(f"skipped C1: {len(imgs)} images in the cell, {len(names)} expected")
            continue
        for o, n in zip(imgs, names):
            o["data"]["image/png"] = base64.b64encode((U.FIG_DIR / n).read_bytes()).decode("ascii") + "\n"
        print(f"C1: {len(names)} embedded images replaced")
        done += len(names)
        continue
    m = re.search(r'U\.fig_name\("(C[23])",\s*"(\w+)",\s*"(\w+)"', src)
    if not m:
        continue
    png = U.FIG_DIR / U.fig_name(m.group(1), m.group(2), m.group(3))
    imgs = [o for o in cell.get("outputs", []) if "data" in o and "image/png" in o["data"]]
    if not png.exists() or len(imgs) != 1:
        print(f"skipped {m.group(1)}: file exists={png.exists()}, images in cell={len(imgs)}")
        continue
    imgs[0]["data"]["image/png"] = base64.b64encode(png.read_bytes()).decode("ascii") + "\n"
    print(f"{m.group(1)}: embedded image replaced by {png.name}")
    done += 1
if done:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)
print(f"{done} image(s) refreshed in {path.name}")
