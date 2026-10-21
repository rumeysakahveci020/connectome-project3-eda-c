"""Rebuild everything from scratch, in the right order, and verify it.

    python tools/rebuild_all.py --dry-run          # show the plan, run nothing
    python tools/rebuild_all.py                    # asks once before deleting the old outputs
    python tools/rebuild_all.py --yes              # no questions
    python tools/rebuild_all.py --skip-build       # keep data/cache; rerun notebooks, summary, verification only

Order: preflight -> backup -> clean -> tests -> FlyWire cache -> male CNS cache (group region rule) -> region rule check
       -> 00_integration -> eda_C (100 null graphs) -> summary from the numbers -> verification.
It stops at the first failure and says which step failed. Everything is also written to logs/rebuild_<time>.log.
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--yes", action="store_true")
ap.add_argument("--skip-build", action="store_true")
ap.add_argument("--seeds", type=int, default=100)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--c6-max-minutes", type=float, default=35)
args = ap.parse_args()
DRY = args.dry_run
STAMP = time.strftime("%Y%m%d_%H%M%S")
LOG = ROOT / "logs" / f"rebuild_{STAMP}.log"
PY = sys.executable
T0 = time.time()

RAW_FILES = {
    "FlyWire edges (Zenodo, Buhmann)": RAW / "proofread_connections_783.feather",
    "FlyWire annotations v3.2.0": RAW / "Supplemental_file1_neuron_annotations.tsv",
    "FlyWire proofread ids": RAW / "proofread_root_ids_783.npy",
    "male CNS edges (connectome-weights)": RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


def say(msg):
    print(msg, flush=True)
    if not DRY:
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


def fail(step, msg):
    say(f"\nSTOPPED at step '{step}': {msg}")
    say(f"Log: {LOG}")
    sys.exit(1)


def run(step, cmd, env=None):
    say(f"\n[{time.time() - T0:6.0f}s] {step}\n$ " + " ".join(str(c) for c in cmd))
    if DRY:
        return
    e = dict(os.environ if env is None else env)
    e["PYTHONIOENCODING"] = "utf-8"
    with open(LOG, "a", encoding="utf-8") as fh:
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        rc = p.wait()
    if rc:
        fail(step, f"exit code {rc}")


# ------------------------------------------------------------------ 0. preflight
say("== 0. preflight")
problems = []
if sys.version_info < (3, 10):
    problems.append(f"Python {sys.version.split()[0]} is too old (need 3.10 or newer)")
for mod in ("pandas", "pyarrow", "matplotlib", "seaborn", "networkx", "nbformat", "nbclient", "pytest"):
    try:
        importlib.import_module(mod)
    except ImportError:
        problems.append(f"missing package '{mod}': python -m pip install -r requirements.txt")
try:
    importlib.import_module("igraph")
except ImportError:
    say("WARN: igraph is not installed; C6 will be much slower (about 8 times in my test) (python -m pip install python-igraph)")
for rel in ("config.py", "eda_shared/eda_helpers.py", "src/build_cache.py", "notebooks/eda_C.ipynb", "tools/verify_outputs.py"):
    if not (ROOT / rel).exists():
        problems.append(f"missing {rel}")
# the code must be the newest version (an old copy would reproduce the old mistakes)
MARKERS = [("src/eda_c.py", "hue_order", "C2 colours (same colour per transmitter as in C1)"),
           ("src/fetch_malecns.py", "roi_connection_counts", "male CNS region = upstream + downstream (group decision)"),
           ("src/eda_utils.py", "GROUP_ROWS", "two group rows for the unspecified ROIs"),
           ("notebooks/eda_C.ipynb", "Threshold:", "figure captions that state the threshold")]
for rel, marker, why in MARKERS:
    path = ROOT / rel
    if path.exists() and marker not in path.read_text(encoding="utf-8", errors="ignore"):
        problems.append(f"{rel} is an old version ({why}); unpack the newest project3_workstream_C.zip over it (not data, .env, eda_shared)")
if not args.skip_build:
    for label, path in RAW_FILES.items():
        if not path.exists():
            problems.append(f"missing raw file: {label}: {path}")
    pkl = RAW / "neuprint_malecns_v1.0_traced.pkl"
    if not pkl.exists() and not (ROOT / ".env").exists():
        problems.append("no saved neuPrint answer and no .env with NEUPRINT_TOKEN")
for p_ in problems:
    say(f"PROBLEM: {p_}")
if problems and not DRY:
    fail("preflight", f"{len(problems)} problem(s) above")
if not problems:
    say("preflight ok")

# ------------------------------------------------------------------ 1. backup and clean
say("\n== 1. backup and clean")
backup = ROOT.parent / f"project3_yedek_{STAMP}"
to_clean = [ROOT / "figures", ROOT / "reports"] + ([] if args.skip_build else [ROOT / "data" / "cache"])
say(f"old figures, reports and notebooks are copied to {backup}")
say("then deleted: " + ", ".join(str(p.relative_to(ROOT)) for p in to_clean))
if not DRY:
    if not args.yes and input("Continue? (y/n) ").strip().lower() != "y":
        sys.exit("cancelled")
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "reports", "notebooks"):
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, backup / name)
    for p_ in to_clean:
        shutil.rmtree(p_, ignore_errors=True)

# ------------------------------------------------------------------ 2. clean notebooks, tests
run("2. tests (project and eda_shared)", [PY, "-m", "pytest", "tests", "eda_shared/tests", "-q"])

# ------------------------------------------------------------------ 3. caches
if not args.skip_build:
    run("3a. FlyWire cache (proofread list, group provisional maps)",
        [PY, "-m", "src.build_cache", "--only", "fafb", "--force", "--fafb-edges", RAW_FILES["FlyWire edges (Zenodo, Buhmann)"],
         "--fafb-annotations", RAW_FILES["FlyWire annotations v3.2.0"], "--fafb-proofread-ids", RAW_FILES["FlyWire proofread ids"],
         "--provisional-maps"])
    run("3b. male CNS cache (region = upstream + downstream, group decision D15)",
        [PY, "-m", "src.build_cache", "--only", "malecns", "--force", "--malecns-edges", RAW_FILES["male CNS edges (connectome-weights)"],
         "--provisional-maps"])
    run("3c. does the saved neuPrint answer carry upstream and downstream?", [PY, "tools/check_roiinfo_rule.py"])
    run("3d. FAFB neurons without a region (UNASGD), for the group's note", [PY, "tools/check_unassigned.py"])

# ------------------------------------------------------------------ 4. notebooks
env = dict(os.environ)
env.pop("EDA_UNMAPPED_REGION", None)           # not needed any more: the two group rows cover the unspecified ROIs
env.update({"EDA_N_SEEDS": str(args.seeds), "EDA_WORKERS": str(args.workers), "EDA_C6_MAX_MINUTES": str(args.c6_max_minutes)})
run("4a. 00_integration (validators, assert_interchangeable, Step 0)", [PY, "tools/run_notebook.py", "notebooks/00_integration.ipynb"], env)
run(f"4b. eda_C (C1-C6, {args.seeds} null graphs; the long step, about 25 minutes)", [PY, "tools/run_notebook.py", "notebooks/eda_C.ipynb"], env)

# ------------------------------------------------------------------ 5. summary and verification
run("5a. summary from the numbers (no number is typed by hand)", [PY, "tools/make_summary.py"])
run("5b. verification (recomputes every number, tests colours, checks notebooks)", [PY, "tools/verify_outputs.py", "--expect-seeds", str(args.seeds)])

say(f"\nDONE in {(time.time() - T0) / 60:.0f} minutes. Old outputs are in {backup}.")
say("Now: (1) open the figures yourself, (2) rewrite the 'What surprised me' sentence in reports/eda_C.md in your own words,")
say("(3) paste reports/eda_C_numbers.json to Claude so the talk and the guides get the new numbers, (4) zip and upload.")
