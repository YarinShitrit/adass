"""Secrets from a `.env` file, with a fallback chain that works everywhere this runs.

`.env` is gitignored. `.env.example` is committed and lists the keys.

**The one thing worth understanding before relying on this.** On Colab the repo arrives by
`git clone`, and `.env` is gitignored -- so it is *not in the clone*. A `.env` at the repo root
therefore cannot supply the GitHub token that performs the clone, and cannot supply anything at
all on a freshly cloned runtime. That is not a bug to work around; it is what gitignoring a
secret means.

So `find_dotenv` searches beyond the repo, and the useful Colab pattern is to keep one `.env` on
Drive (`/content/drive/MyDrive/.env` or `MyDrive/adass/.env`), where it persists across runtimes
and is never committed. Failing that, `require()` prompts.

Resolution order, first hit wins:

    1. the process environment          -- set by you, or by an earlier call here
    2. a .env file                      -- repo root, cwd, ~, /content, Drive
    3. Colab Secrets                    -- browser Colab frontend only
    4. an interactive getpass prompt    -- works everywhere, stores nothing
"""
import os
from pathlib import Path

from . import paths

#: Keys this project uses. `.env.example` documents each one.
KEYS = ("HF_TOKEN", "GH_TOKEN", "ANTHROPIC_API_KEY")

_CANDIDATES = (
    lambda: paths.ROOT / ".env",
    lambda: Path.cwd() / ".env",
    lambda: Path("/content/drive/MyDrive/adass/.env"),
    lambda: Path("/content/drive/MyDrive/.env"),
    lambda: Path("/content/.env"),
    lambda: Path.home() / ".env",
)


def find_dotenv():
    """First existing .env among the candidate locations, or None."""
    for get in _CANDIDATES:
        try:
            p = get()
        except Exception:
            continue
        if p.is_file():
            return p
    return None


def parse_dotenv(path):
    """Minimal KEY=VALUE parser: comments, blank lines, `export` prefix, quoted values.

    Deliberately dependency-free rather than python-dotenv: this has to be importable on a
    fresh Colab runtime before anything is installed, and the format is eight lines of code.
    """
    out = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_env(path=None, override=False, verbose=True):
    """Load a .env into os.environ. Existing variables win unless `override`.

    Returns the list of keys set. Values are never printed -- only key names.
    """
    p = Path(path) if path else find_dotenv()
    if p is None or not Path(p).is_file():
        if verbose:
            print("no .env found (searched repo root, cwd, Drive, home) -- "
                  "using the environment, Colab Secrets, or a prompt")
        return []
    loaded = []
    for k, v in parse_dotenv(p).items():
        if override or not os.environ.get(k):
            os.environ[k] = v
            loaded.append(k)
    if verbose:
        print(f"loaded {len(loaded)} key(s) from {p}: {', '.join(loaded) or '(none new)'}")
    return loaded


def require(name, prompt=None, allow_prompt=True):
    """Get a secret, or raise. Never writes the value anywhere but os.environ."""
    if os.environ.get(name):
        return os.environ[name]

    load_env(verbose=False)
    if os.environ.get(name):
        return os.environ[name]

    try:                                            # browser Colab only
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            os.environ[name] = v
            return v
    except Exception:
        pass

    if allow_prompt:
        import getpass
        v = getpass.getpass(prompt or f"{name}: ")
        if v:
            os.environ[name] = v
            return v

    raise RuntimeError(
        f"{name} is not set. Put it in a .env (see .env.example), export it, "
        f"or add it to Colab Secrets.")


def status():
    """Which keys are present, without revealing any of them."""
    p = find_dotenv()
    lines = [f".env: {p or 'not found'}"]
    for k in KEYS:
        v = os.environ.get(k)
        lines.append(f"  {k:20} {'set (' + str(len(v)) + ' chars)' if v else 'not set'}")
    return "\n".join(lines)
