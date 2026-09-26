"""Portable read-only, network-isolated Python worker command."""
from pathlib import Path
import shutil
import sys


def sandbox_command(module="verl.utils.reward_score.code_grader_worker"):
    for executable in ("unshare", "bwrap"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Install the required sandbox executable: {executable}")
    runtime = Path(__file__).resolve().parents[2]
    command = ["unshare", "--user", "--map-root-user", "--net", "bwrap",
               "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts"]
    mounts = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64", sys.prefix, sys.base_prefix)] + [runtime]
    seen = set()
    for path in mounts:
        if path.exists() and str(path) not in seen:
            command += ["--ro-bind", str(path), str(path)]
            seen.add(str(path))
    command += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--clearenv",
                "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONPATH", str(runtime),
                "--setenv", "PYTHONNOUSERSITE", "1", "--setenv", "OMP_NUM_THREADS", "1",
                "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "MKL_NUM_THREADS", "1",
                sys.executable, "-m", module]
    return command
