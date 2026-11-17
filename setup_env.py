"""Set up the Python side of InfiniSplat v3 on the Windows lab PC (no admin).

Runs inside the activated conda env (setup.bat creates and activates it).
Stdlib only at top level, torch is never imported here: every torch check runs
in a fresh interpreter. Every step checks first and only does what is missing,
so reruns are fast.

Usage: python setup_env.py

Steps: 2 git, 3 clone, 4 torch, 5 requirements, 6 weights, 7 verify
(step 1 is the conda env, done by setup.bat).

Follows the official InfiniSplat_v3/README.md installation (Python 3.10+, PyTorch
2.4+ with a CUDA build that fits the driver installed first, then
python -m pip install -r requirements.txt inside InfiniSplat_v3), the weights
location InfiniSplat_v3/checkpoints/ from the same README, and the unit tests
of docs/inference.md (python -m unittest discover -s tests -v). InfiniSplat v3
compiles nothing, so there is no compiler or CUDA toolkit step.

Deviations, each forced by this PC or by the lab setup:
  conda env              an isolated env, because there is no admin and no system Python
  Python 3.11            inside the 3.10+ range, and what the env is created with
  torch 2.9.1 cu128      the RTX 5070 is sm_120, which needs a CUDA 12.8 build of torch
  git clone core.longpaths   Windows path length limit, pinned to one commit
  weights by curl        from pinned Hugging Face URLs, resumable, with sha256 checks
  unit tests             run with `validation` pointing at inference.py, one stale test allowed
                         to fail (both are out of date upstream, see KNOWN_UPSTREAM_TEST_FAILURES)
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path





# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_NAME = "infinisplat3"
PYTHON_VERSION = "3.11"
TOTAL_STEPS = 7

REPO_URL = "https://github.com/zju3dv/InfiniSplat.git"
REPO_COMMIT = "f2140ecdc94d38691804d689cce412858980a7bc"
REPO_FOLDER = "InfiniSplat"
V3_FOLDER = "InfiniSplat_v3"

CU128_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION = "2.9.1"
TORCH_CUDA = "12.8"
REQUIRED_ARCH = "sm_120"

REQUIREMENTS_REL = "requirements.txt"
REQUIREMENTS_MARKER = "infinisplat_v3_requirements.sha256"

HF_BASE_URL = (
    "https://huggingface.co/ritianyu/SplatAnything/resolve/"
    "cea57139b63da978424b5faca31e06573261215e"
)
CHECKPOINT_REL = "checkpoints"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MIN_FREE_GB = 30

# name: file name on Hugging Face and in checkpoints/. size in bytes and sha256 as listed there.
WEIGHTS = [
    {
        "name": "SplatAnything.ckpt",
        "size": 11334600799,
        "sha256": "acbb2544cc213a78ecad3ad1f4b88f4a751831a056433331738c30d7b13b3c5c",
    },
    {
        "name": "worldmirror2.safetensors",
        "size": 5053556568,
        "sha256": "0e37dcf97576b1d8d9ef16faf3f6fb57bfd1d02e287255d5540117e229077795",
    },
    {
        # Optional in the official docs, used only with --sky_seg. Small, so it is downloaded anyway.
        "name": "skyseg.onnx",
        "size": 175997079,
        "sha256": "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39",
    },
]

# The official test command is python -m unittest discover -s tests -v (docs/inference.md).
# At the pinned commit two tests are out of date upstream, on every OS:
#   tests/test_inference.py imports `validation`, the old name of inference.py
#   test_documented_camera_json_matches_loader reads prepare_data.md from the v3 root, it is in docs/
# So the tests run in one interpreter with `validation` pointing at inference.py, and the
# second test is the only failure allowed.
KNOWN_UPSTREAM_TEST_FAILURES = {"test_documented_camera_json_matches_loader"}
UNIT_TEST_CODE = (
    "import sys, unittest\n"
    "sys.path[:0] = ['.', 'tests']\n"
    "import inference\n"
    "sys.modules['validation'] = inference\n"
    "suite = unittest.defaultTestLoader.discover('tests')\n"
    "result = unittest.TextTestRunner(verbosity=2).run(suite)\n"
    f"known = {sorted(KNOWN_UPSTREAM_TEST_FAILURES)!r}\n"
    "bad = [t for t, _ in result.failures + result.errors if t.id().split('.')[-1] not in known]\n"
    "print('tests run:', result.testsRun, '| failed:', len(result.failures) + len(result.errors),"
    " '| failed outside the known upstream ones:', len(bad))\n"
    "sys.exit(1 if bad else 0)\n"
)
IMPORT_CHECK_CODE = (
    "import inference\n"
    "from src.models.models.gaussian_worldmirror import GaussianWorldMirror\n"
    "from src.models.models.geometry_worldmirror import GeometryWorldMirror\n"
    "print('inference, GaussianWorldMirror, GeometryWorldMirror import')\n"
)
SMI_QUERY_ARGS = ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / REPO_FOLDER
V3_DIR = REPO_DIR / V3_FOLDER
CHECKPOINT_DIR = V3_DIR / CHECKPOINT_REL





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(i, total, title):
    """Print a step header like [2/7] Title.

    Args:
        i: Step number.
        total: Total number of steps.
        title: Step title.

    Returns:
        None.
    """
    print()
    print(f"[{i}/{total}] {title}")



def fail(msg):
    """Print an error message and stop with exit code 1.

    Args:
        msg: What went wrong and what to do about it.

    Returns:
        Never returns.
    """
    print(f"ERROR: {msg}")
    sys.exit(1)



def pip(*args, cwd=None):
    """Run python -m pip with this interpreter and show its output.

    Args:
        *args: Arguments passed to pip.
        cwd: Working directory, or None for the current one.

    Returns:
        None. Raises CalledProcessError when pip fails.
    """
    cmd = [sys.executable, "-m", "pip", *args]
    print("  >", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)



def pip_or_fail(*args, cwd=None):
    """Run pip and stop the setup with a clear message when it fails.

    Args:
        *args: Arguments passed to pip.
        cwd: Working directory, or None for the current one.

    Returns:
        None.
    """
    try:
        pip(*args, cwd=cwd)
    except subprocess.CalledProcessError:
        fail("pip failed (see output above). Send the last 40 lines of this window.")



def run_py(code, cwd=None, capture=True):
    """Run code in a fresh interpreter so a broken import cannot kill this script.

    Args:
        code: Python source passed to python -c.
        cwd: Working directory, or None for the current one.
        capture: True to capture stdout and stderr, False to show them live.

    Returns:
        The CompletedProcess.
    """
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, capture_output=capture, text=True,
        errors="replace",
    )



def sha256_of_file(path):
    """Hash a file in chunks, so an 11 GB file does not have to fit in memory.

    Args:
        path: File to hash.

    Returns:
        The lowercase sha256 hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()



def conda_exe():
    """Find the conda executable.

    Args:
        None.

    Returns:
        Path of conda as a string. Stops the setup when it is not found.
    """
    exe = os.environ.get("CONDA_EXE")
    if exe and Path(exe).exists():
        return exe
    found = shutil.which("conda")
    if not found:
        fail("conda not found (CONDA_EXE is not set). Run this through setup.bat.")
    return found



def conda_install(*pkgs):
    """Install packages into this env from conda-forge only.

    Args:
        *pkgs: Package specs.

    Returns:
        True when conda succeeded.
    """
    cmd = [conda_exe(), "install", "-y", "-p", sys.prefix,
           "-c", "conda-forge", "--override-channels", *pkgs]
    print("  >", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0



def prepend_path(var, dirs):
    """Put existing dirs at the front of a semicolon separated variable, without duplicates.

    Args:
        var: Environment variable name, such as PATH.
        dirs: Folders to add when they exist and are not there yet.

    Returns:
        None.
    """
    parts = [p for p in os.environ.get(var, "").split(";") if p]
    have = {os.path.normcase(os.path.normpath(p)) for p in parts}
    new = []
    for d in dirs:
        d = Path(d)
        if d.is_dir() and os.path.normcase(os.path.normpath(str(d))) not in have:
            new.append(str(d))
    if new:
        os.environ[var] = ";".join(new + parts)



def lib_dir():
    """The conda Library folder, where conda puts git on Windows.

    Args:
        None.

    Returns:
        Path of <env>\\Library.
    """
    return Path(sys.prefix) / "Library"



def report_nvidia_smi():
    """Print the GPU, driver and memory line from nvidia-smi. Never fails the setup.

    Args:
        None.

    Returns:
        None.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        print("  nvidia-smi not found, skipping the driver line")
        return
    try:
        r = subprocess.run([exe, *SMI_QUERY_ARGS], capture_output=True, text=True, errors="replace")
    except OSError:
        print("  nvidia-smi could not be run, skipping the driver line")
        return
    line = r.stdout.strip().splitlines()[:1]
    if r.returncode == 0 and line:
        print(f"  GPU, driver, memory: {line[0]}")
    else:
        print("  nvidia-smi gave no GPU line, skipping the driver line")





# ---------------------------------------------------------------------------
# Step 2: git
# ---------------------------------------------------------------------------

def ensure_git():
    """Make sure git is on PATH, installing it from conda-forge when missing.

    Args:
        None.

    Returns:
        None.
    """
    if shutil.which("git"):
        print("  OK, already done")
        return
    print("  git not found, installing it from conda-forge")
    if not conda_install("git"):
        fail("conda install git failed (see output above)")
    prepend_path("PATH", [lib_dir() / "bin", lib_dir() / "cmd"])
    if shutil.which("git") is None:
        fail("git is still not on PATH after installing it")





# ---------------------------------------------------------------------------
# Step 3: clone
# ---------------------------------------------------------------------------

def git_run(repo, *args):
    """Run git quietly inside a repo and capture its output.

    Args:
        repo: Folder to run in.
        *args: git arguments.

    Returns:
        The CompletedProcess.
    """
    # Stop git from looking above the repo, so a repository around this folder
    # cannot make status or rev-parse answer for the wrong repo.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=Path(repo).parent.as_posix())
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, errors="replace"
    )



def git_show(cwd, *args):
    """Run git with its output on the console, for network steps.

    Args:
        cwd: Folder to run in.
        *args: git arguments.

    Returns:
        The exit code.
    """
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    cmd = ["git", *args]
    print("  >", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env).returncode



def git_head(repo):
    """Commit that HEAD points to.

    Args:
        repo: Clone folder.

    Returns:
        The full commit hash, or None when git could not answer.
    """
    r = git_run(repo, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None



def clone_repo():
    """Clone InfiniSplat and pin it to REPO_COMMIT, or check an existing clone.

    An existing clone at another commit is checked out when it has no local
    changes, and refused when it has some.

    Args:
        None.

    Returns:
        The clone folder.
    """
    if shutil.which("git") is None:
        fail("git not found on PATH (step 2 should have installed it)")
    dest = REPO_DIR
    if dest.exists():
        if not (dest / ".git").exists():
            fail(f"{dest} exists but is not a git clone. Remove or rename it and run again.")
        head = git_head(dest)
        if head == REPO_COMMIT:
            print(f"  OK, {dest.name} is already at {REPO_COMMIT[:7]}")
        else:
            dirty = git_run(dest, "status", "--porcelain").stdout.strip()
            if dirty:
                fail(f"{dest} is at {head} with local changes, expected {REPO_COMMIT}. "
                     "Remove or rename it and run again.")
            print(f"  {dest.name} is at {str(head)[:7]}, checking out {REPO_COMMIT[:7]}")
            if git_show(dest, "checkout", REPO_COMMIT) != 0:
                fail(f"git checkout {REPO_COMMIT} failed in {dest}")
    else:
        rc = git_show(BASE_DIR, "clone", "-c", "core.longpaths=true", REPO_URL, str(dest))
        if rc != 0:
            fail(f"git clone failed (see output above). Remove {dest} if it was left behind.")
        if git_show(dest, "checkout", REPO_COMMIT) != 0:
            fail(f"git checkout {REPO_COMMIT} failed in {dest}")
    if git_head(dest) != REPO_COMMIT:
        fail(f"{dest} is not at {REPO_COMMIT} after checkout")
    if not (V3_DIR / REQUIREMENTS_REL).is_file():
        fail(f"{V3_DIR / REQUIREMENTS_REL} not found (upstream layout changed?)")
    print(f"  {V3_FOLDER} found, {dest.name} at {REPO_COMMIT[:7]}")
    return dest





# ---------------------------------------------------------------------------
# Step 4: torch
# ---------------------------------------------------------------------------

def torch_is_expected():
    """Check in a fresh interpreter that torch is TORCH_VERSION built for TORCH_CUDA.

    Args:
        None.

    Returns:
        True when the installed torch is the expected build.
    """
    code = (
        "import sys, torch\n"
        f"ok = torch.__version__.startswith('{TORCH_VERSION}') and torch.version.cuda == '{TORCH_CUDA}'\n"
        "print(torch.__version__, 'cuda', torch.version.cuda)\n"
        "sys.exit(0 if ok else 1)\n"
    )
    r = run_py(code, cwd=sys.prefix)
    if r.stdout.strip():
        print("  found:", r.stdout.strip())
    return r.returncode == 0



def install_torch(force=False):
    """Install the pinned torch from the cu128 index and check the result.

    Only torch, no torchvision: InfiniSplat v3 does not use it.

    Args:
        force: True to add --force-reinstall --no-deps, used after pip swapped torch.

    Returns:
        None.
    """
    flags = ["--force-reinstall", "--no-deps"] if force else []
    pip_or_fail("install", *flags, f"torch=={TORCH_VERSION}", "--index-url", CU128_INDEX)
    if not torch_is_expected():
        fail(f"torch is still not the {TORCH_VERSION} CUDA {TORCH_CUDA} build after installing")



def ensure_torch():
    """Install torch unless the expected build is already there.

    Args:
        None.

    Returns:
        None.
    """
    if torch_is_expected():
        print("  OK, already done")
        return
    print(f"  installing the torch {TORCH_VERSION} cu128 build")
    install_torch()



def require_cuda_and_arch():
    """Require REQUIRED_ARCH in the torch build and a usable GPU, and print the GPU.

    Args:
        None.

    Returns:
        None. Stops the setup when a requirement is not met.
    """
    code = (
        "import sys, torch\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
        "arch = torch.cuda.get_arch_list()\n"
        "print('arch list:', ' '.join(arch))\n"
        "avail = torch.cuda.is_available()\n"
        "print('torch.cuda.is_available():', avail)\n"
        f"if '{REQUIRED_ARCH}' not in arch:\n"
        f"    print('{REQUIRED_ARCH} (RTX 50 series) is missing from this torch build')\n"
        "    sys.exit(2)\n"
        "if not avail:\n"
        "    sys.exit(1)\n"
        "p = torch.cuda.get_device_properties(0)\n"
        "print('GPU:', p.name, '|', round(p.total_memory / 1024**3, 1), 'GB VRAM')\n"
    )
    r = run_py(code, cwd=sys.prefix, capture=False)
    if r.returncode == 2:
        fail(f"this torch build has no {REQUIRED_ARCH} kernels (see output above)")
    if r.returncode != 0:
        fail("CUDA is not available to torch (see output above)")





# ---------------------------------------------------------------------------
# Step 5: requirements
# ---------------------------------------------------------------------------

def requirements_done(req_path):
    """Check whether this exact requirements file was already installed.

    Args:
        req_path: Path of requirements.txt.

    Returns:
        True when the stored hash in the env matches the file.
    """
    marker = Path(sys.prefix) / REQUIREMENTS_MARKER
    return marker.exists() and marker.read_text().strip() == sha256_of_file(req_path)



def mark_requirements_done(req_path):
    """Store the requirements file hash in the env.

    Args:
        req_path: Path of requirements.txt.

    Returns:
        None.
    """
    marker = Path(sys.prefix) / REQUIREMENTS_MARKER
    marker.write_text(sha256_of_file(req_path) + "\n")





# ---------------------------------------------------------------------------
# Step 6: weights
# ---------------------------------------------------------------------------

def read_marker(marker):
    """Read the hash stored in a marker file.

    Args:
        marker: Path of the .sha256 marker.

    Returns:
        The stored hash, or an empty string when there is no marker.
    """
    if not marker.is_file():
        return ""
    return marker.read_text().strip()



def discard_weight(dest, marker):
    """Delete a weight file and its marker.

    Args:
        dest: Weight file.
        marker: Its .sha256 marker.

    Returns:
        None.
    """
    dest.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)



def download_weight(url, dest):
    """Download or resume a file with curl.

    Args:
        url: Source URL.
        dest: Target file. A partial file is continued with -C -.

    Returns:
        None. Stops the setup when curl fails.
    """
    cmd = ["curl.exe", "-L", "--fail", "-C", "-", "-o", str(dest), url]
    print("  >", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        fail(f"download failed: {url}\n  Run setup.bat again, the download resumes where it stopped.")



def ensure_weight(weight):
    """Make one weight file present with the right size and sha256.

    A file is done when its size matches and its marker holds the expected hash.
    A file with the right size but no marker is hashed once and marked. A file
    that is too large or has the wrong hash is deleted and the setup stops.

    Args:
        weight: One entry of WEIGHTS.

    Returns:
        None.
    """
    name, size, sha = weight["name"], weight["size"], weight["sha256"]
    dest = CHECKPOINT_DIR / name
    marker = dest.with_name(name + ".sha256")
    url = f"{HF_BASE_URL}/{name}"

    if dest.is_file() and dest.stat().st_size > size:
        discard_weight(dest, marker)
        fail(f"{name} was larger than the expected {size} bytes. The file was deleted, run again.")
    if dest.is_file() and dest.stat().st_size == size and read_marker(marker) == sha:
        print(f"  {name}: OK, already done")
        return

    if not (dest.is_file() and dest.stat().st_size == size):
        marker.unlink(missing_ok=True)
        have = dest.stat().st_size if dest.is_file() else 0
        print(f"  {name}: downloading ({have} of {size} bytes present)")
        download_weight(url, dest)
        got_size = dest.stat().st_size if dest.is_file() else 0
        if got_size > size:
            discard_weight(dest, marker)
            fail(f"{name} is larger than the expected {size} bytes. The file was deleted, run again.")
        if got_size < size:
            fail(f"{name} is incomplete ({got_size} of {size} bytes). Run setup.bat again to resume.")

    print(f"  {name}: checking sha256 (one time, can take a minute for large files)")
    got = sha256_of_file(dest)
    if got != sha:
        discard_weight(dest, marker)
        fail(f"sha256 mismatch for {name}\n  expected {sha}\n  got      {got}\n"
             "  The file and its marker were deleted. Run setup.bat again to download it fresh.")
    marker.write_text(sha + "\n")
    print(f"  {name}: hash OK")



def ensure_weights():
    """Download and verify every weight file into checkpoints/.

    Args:
        None.

    Returns:
        None.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    needs_download = [
        w for w in WEIGHTS
        if not (CHECKPOINT_DIR / w["name"]).is_file()
        or (CHECKPOINT_DIR / w["name"]).stat().st_size < w["size"]
    ]
    if needs_download:
        total_gb = sum(w["size"] for w in WEIGHTS) / 1e9
        print(f"  The weights total about {total_gb:.1f} GB. A stopped download resumes on the next run.")
        free_gb = shutil.disk_usage(CHECKPOINT_DIR).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"  WARNING: only {free_gb:.0f} GB free on this drive, want at least {MIN_FREE_GB} GB.")
    for weight in WEIGHTS:
        ensure_weight(weight)





# ---------------------------------------------------------------------------
# Step 7: verify
# ---------------------------------------------------------------------------

def run_checks():
    """Run the official unit tests and the import check, both with cwd InfiniSplat_v3.

    Args:
        None.

    Returns:
        List of labels of the checks that failed, empty when all passed.
    """
    failed = []

    print("  running the unit tests (unittest discover on tests, see KNOWN_UPSTREAM_TEST_FAILURES)")
    r = subprocess.run([sys.executable, "-c", UNIT_TEST_CODE], cwd=V3_DIR)
    if r.returncode == 0:
        print("  OK      unit tests")
    else:
        failed.append("unit tests")
        print("  FAILED  unit tests")

    r = run_py(IMPORT_CHECK_CODE, cwd=V3_DIR)
    if r.returncode == 0:
        detail = r.stdout.strip().splitlines()[-1:]
        print("  OK      import check" + (f"  ({detail[0]})" if detail else ""))
    else:
        failed.append("import check")
        tail = (r.stderr or r.stdout).strip().splitlines()[-4:]
        print("  FAILED  import check")
        for line in tail:
            print(f"          {line}")
    return failed





# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def setup():
    """Run steps 2 to 7 in order.

    Args:
        None.

    Returns:
        None.
    """
    total = TOTAL_STEPS
    if os.name != "nt":
        fail("this setup is for Windows only")
    if Path(sys.prefix).name != ENV_NAME:
        fail(f"this is not the conda env {ENV_NAME} ({sys.prefix}). Run this through setup.bat.")
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    prepend_path("PATH", [lib_dir() / "bin", lib_dir() / "cmd"])

    step(2, total, "git")
    ensure_git()

    step(3, total, f"Clone InfiniSplat (commit {REPO_COMMIT[:7]})")
    clone_repo()

    step(4, total, f"torch {TORCH_VERSION}, CUDA {TORCH_CUDA} build")
    report_nvidia_smi()
    ensure_torch()
    require_cuda_and_arch()

    step(5, total, f"{V3_FOLDER} requirements (pip install -r {REQUIREMENTS_REL})")
    req_path = V3_DIR / REQUIREMENTS_REL
    if requirements_done(req_path):
        print("  OK, already done")
    else:
        pip_or_fail("install", "-r", str(req_path), cwd=V3_DIR)
        if torch_is_expected():
            print("  OK, torch is still the cu128 build")
        else:
            print("  torch was swapped by pip, reinstalling the cu128 build")
            install_torch(force=True)
            require_cuda_and_arch()
        mark_requirements_done(req_path)

    step(6, total, f"Weights into {V3_FOLDER}\\{CHECKPOINT_REL}")
    ensure_weights()

    step(7, total, "Verify")
    failed = run_checks()
    if failed:
        fail("verification failed for: " + ", ".join(failed))
    print("INFINISPLAT V3 ENV OK")



def main():
    """Print where this runs and run the setup.

    Args:
        None.

    Returns:
        None.
    """
    print(f"Python {sys.version.split()[0]} in {sys.prefix}")
    print(f"InfiniSplat v3 into env {ENV_NAME} (python {PYTHON_VERSION})")
    setup()
    print()
    print("Setup finished.")



if __name__ == "__main__":
    main()
