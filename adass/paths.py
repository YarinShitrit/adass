"""Where things live, resolved from the repo rather than from the working directory.

Every notebook used to open artifacts by bare filename, so it only worked if the kernel
happened to start in the project root -- and once notebooks moved into `notebooks/`, none of
them would have. Paths are resolved here instead, once.

Override with the `ADASS_ROOT` environment variable if you keep data elsewhere.
"""
import os
from pathlib import Path


def _detect_root() -> Path:
    env = os.environ.get("ADASS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[1]          # editable install / repo checkout
    if (here / "data").is_dir():
        return here
    cur = Path.cwd().resolve()                          # installed elsewhere: walk up from cwd
    for cand in (cur, *cur.parents):
        if (cand / "pyproject.toml").is_file() and (cand / "data").is_dir():
            return cand
    return here


ROOT        = _detect_root()
DATA        = ROOT / "data"
GOLD        = DATA / "gold"
GENERATIONS = DATA / "generations"
VECTORS     = DATA / "vectors"
RESULTS     = DATA / "results"
FIGURES     = ROOT / "figures"
CONFIG_DIR  = ROOT / "config"
DOCS        = ROOT / "docs"
NOTEBOOKS   = ROOT / "notebooks"

_SEARCH = (RESULTS, GENERATIONS, GOLD, VECTORS, CONFIG_DIR, FIGURES, ROOT)


def artifact(name, must_exist=True) -> Path:
    """Resolve a bare artifact filename to its path. Absolute paths pass through."""
    p = Path(name)
    if p.is_absolute():
        return p
    for d in _SEARCH:
        cand = d / p
        if cand.exists():
            return cand
    if must_exist:
        raise FileNotFoundError(
            f"{name!r} not found under {ROOT}. Searched: "
            + ", ".join(str(d.relative_to(ROOT)) for d in _SEARCH))
    return RESULTS / p


def results_path(name) -> Path:
    """Where a results file is written. Bare names land in data/results."""
    p = Path(name)
    return p if p.is_absolute() or len(p.parts) > 1 else RESULTS / p


def figure(name) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name


def config() -> Path:
    return CONFIG_DIR / "adass_config.json"


def describe() -> str:
    return (f"repo root  {ROOT}\n"
            f"data       {DATA}  ({'present' if DATA.is_dir() else 'MISSING'})\n"
            f"results    {RESULTS}")
