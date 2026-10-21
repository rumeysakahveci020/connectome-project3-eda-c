"""Last check before uploading to UZEM and GitHub. Prints PASS / WARN / FAIL and exits 1 on any FAIL.

    python tools/final_check.py                       # everything, including tools/verify_outputs.py (about 4 minutes)
    python tools/final_check.py --skip-verify         # the quick checks only
    python tools/final_check.py --zip project3_eda_C.zip --worksheet "C:\\path\\Decision_Worksheet_Rumeysa.docx"

Checks: the verifier, git (nothing uncommitted, pushed), secrets in the repo and in the zip, the zip against the working
folder (byte for byte), the one-page report, the README numbers, the HTML export, the worksheet.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--skip-verify", action="store_true")
ap.add_argument("--zip", default=str(ROOT / "project3_eda_C.zip"))
ap.add_argument("--worksheet", default=None)
args = ap.parse_args()
R = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    R.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def section(t):
    print(f"\n=== {t}", flush=True)


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def norm(b):
    return b.replace(b"\r\n", b"\n")


# ---------------------------------------------------------------- A. verifier
section("A. the verifier")
if args.skip_verify:
    print("skipped (--skip-verify)")
else:
    r = run([sys.executable, "tools/verify_outputs.py", "--zip", args.zip])
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("{'PASS")]
    check("tools/verify_outputs.py: no FAIL", r.returncode == 0, last[-1] if last else r.stdout[-300:])

# ---------------------------------------------------------------- B. git
section("B. git")
inrepo = run(["git", "rev-parse", "--is-inside-work-tree"]).stdout.strip() == "true"
if not inrepo:
    check("this folder is a git repository", False, "run: git init", warn=True)
else:
    dirty = [ln for ln in run(["git", "status", "--porcelain"]).stdout.splitlines() if not ln.startswith("??")]
    check("no uncommitted change in tracked files", not dirty, "; ".join(dirty[:6]) if dirty else "")
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    origin = run(["git", "rev-parse", "origin/main"]).stdout.strip()
    check("HEAD equals origin/main (everything is pushed)", bool(head) and head == origin, f"HEAD {head[:7]}, origin/main {origin[:7] or 'none'}")
    remote = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    check("a remote named origin exists", bool(remote), remote)
    tracked = run(["git", "ls-files"]).stdout.splitlines()
    check("no .env and no data/ in git", not [t for t in tracked if t == ".env" or t.startswith("data/")], "")
    jwt = re.compile(rb"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}")
    leaked = []
    for t in tracked:
        try:
            b = (ROOT / t).read_bytes()
        except OSError:
            continue
        if jwt.search(b) or re.search(rb"NEUPRINT_TOKEN\s*=\s*\S{10,}", b):
            leaked.append(t)
    check("no token in any tracked file", not leaked, f"found in: {leaked}" if leaked else "")

# ---------------------------------------------------------------- C. the zip
section("C. the hand-in zip")
zp = Path(args.zip)
if not zp.exists():
    check("zip exists", False, f"{zp} not found: build it (Compress-Archive ...)")
else:
    z = zipfile.ZipFile(zp)
    names = [n for n in z.namelist() if not n.endswith("/")]
    secrets = [n for n in names if re.search(r"(^|/)\.env$", n) or n.startswith("data/")]
    check("zip has no .env and no data/", not secrets, str(secrets[:4]) if secrets else "")
    jwt = re.compile(rb"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}")
    leaked = [n for n in names if jwt.search(z.read(n)) or re.search(rb"NEUPRINT_TOKEN\s*=\s*\S{10,}", z.read(n))]
    check("no token inside the zip", not leaked, str(leaked[:4]) if leaked else "")
    diff, missing = [], []
    for n in names:
        f = ROOT / n
        if not f.exists():
            missing.append(n)
        elif hashlib.sha256(norm(z.read(n))).digest() != hashlib.sha256(norm(f.read_bytes())).digest():
            diff.append(n)
    check("every file in the zip is identical to the working folder (so the zip is not stale)", not diff, f"different: {diff[:6]}" if diff else "")
    check("no file in the zip is missing from the working folder", not missing, f"{missing[:6]}" if missing else "", warn=True)
    figs = [n for n in names if n.startswith("figures/eda/") and n.endswith(".png")]
    check("zip contains the 14 figures", len(figs) == 14, str(len(figs)))
    need = ["README.md", "reports/eda_C.md", "reports/eda_C_numbers.json", "notebooks/eda_C.ipynb", "notebooks/00_integration.ipynb", "eda_shared/__init__.py"]
    check("zip contains the key files", all(n in names for n in need), str([n for n in need if n not in names]) if any(n not in names for n in need) else "")

# ---------------------------------------------------------------- D. report and README
section("D. report and README")
rep = (ROOT / "reports" / "eda_C.md")
if rep.exists():
    t = rep.read_text(encoding="utf-8")
    check("report is one page (at most 640 words)", len(t.split()) <= 640, f"{len(t.split())} words")
    m = re.search(r"\*\*What surprised me:\*\*\s*(.+)", t)
    check("the 'What surprised me' line exists", bool(m))
    default = bool(m and "looked alike" in m.group(1) and "consensusNt" in m.group(1) and "16-point gap" in m.group(1))
    check("the 'What surprised me' sentence is your own, not the generated default", not default, "still the generated sentence", warn=True)
    check("report has no placeholder text", "TODO" not in t and "[write" not in t)
    numbers = json.load(open(ROOT / "reports" / "eda_C_numbers.json", encoding="utf-8"))
    check("report names the 761 unspecified neurons or the group rule", "unspecified" in t or "761" in t)
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
check("README: region rule, 761 neurons and the mushroom body change are written down",
      all(k in readme for k in ("761", "upstream + downstream", "4,435")), "")
check("README has no stale '182 male CNS'", "182 male CNS" not in readme)
check("exports/eda_C.html exists (optional, for readers without Jupyter)", (ROOT / "exports" / "eda_C.html").exists(), "", warn=True)

# ---------------------------------------------------------------- E. worksheet
section("E. worksheet")
ws = Path(args.worksheet) if args.worksheet else None
if ws is None:
    cands = list(ROOT.glob("Decision_Worksheet*.docx")) + list(Path.home().glob("Downloads/Decision_Worksheet*.docx"))
    ws = max(cands, key=lambda p: p.stat().st_mtime) if cands else None
if ws is None or not ws.exists():
    check("worksheet found", False, "pass --worksheet <path>", warn=True)
else:
    xml = zipfile.ZipFile(ws).read("word/document.xml").decode("utf-8", "ignore")
    text = re.sub(r"<[^>]+>", " ", xml)
    check(f"worksheet {ws.name} holds the male CNS region rule (D15)", "upstream + downstream" in text, str(ws))

print("\n=== SUMMARY")
c = {s: sum(1 for r in R if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
print(c)
for s, n, d in R:
    if s == "FAIL":
        print(f"  FAIL: {n} {d}")
sys.exit(1 if c["FAIL"] else 0)
