from __future__ import annotations

import faulthandler
import gc
import os
import socket
import sys
import time
import traceback
from typing import Callable, Dict, List, Any

from mpi4py import MPI


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _envpick(keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ[k]
    return out


def make_rank_logger(comm: MPI.Comm) -> Callable[[str], None]:
    rank = comm.rank
    size = comm.size
    pid = os.getpid()

    def rprint(msg: str) -> None:
        print(f"[{_now()}] [rank {rank}/{size}] [pid {pid}] {msg}", flush=True)

    return rprint


def print_environment(comm: MPI.Comm, rprint: Callable[[str], None]) -> None:
    import dolfinx

    host = socket.gethostname()

    rprint(f"Host={host}, dolfinx={getattr(dolfinx, '__version__', 'unknown')}")
    rprint(f"MPI library: {MPI.Get_library_version().strip()}")
    rprint(
        "Env threads: "
        f"{_envpick(['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'])}"
    )
    rprint(
        "Env JIT-ish: "
        f"{_envpick([k for k in os.environ.keys() if 'JIT' in k or 'FFCX' in k or 'UFL' in k or 'DOLFINX' in k])}"
    )


def setup_mpi_debug(comm: MPI.Comm) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        comm.Set_errhandler(MPI.ERRORS_RETURN)
    except Exception:
        pass


def setup_faulthandler(*, rprint: Callable[[str], None] | None = None) -> None:
    if os.environ.get("DEBUG_FAULTHANDLER", "1") != "1":
        return
    try:
        faulthandler.enable()
        sec = float(os.environ.get("DEBUG_DUMP_EVERY", "30"))
        faulthandler.dump_traceback_later(sec, repeat=True, file=sys.stderr)
        if rprint is not None:
            rprint(f"faulthandler enabled; will dump tracebacks every {sec}s if hung.")
    except Exception as e:
        if rprint is not None:
            rprint(f"[warning] faulthandler setup failed: {type(e).__name__}: {e}")


def barrier(comm: MPI.Comm, tag: str, rprint: Callable[[str], None] | None = None) -> None:
    if os.environ.get("DEBUG_BARRIERS", "1") != "1":
        return
    if rprint is not None:
        rprint(f"ENTER BARRIER: {tag}")
    comm.Barrier()
    if rprint is not None:
        rprint(f"EXIT  BARRIER: {tag}")


def abort_on_exception(comm: MPI.Comm, rprint: Callable[[str], None], exc: BaseException) -> None:
    rprint(f"!!! EXCEPTION: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    try:
        comm.Abort(1)
    except Exception:
        raise


def close_if_possible(obj: Any) -> None:
    if obj is None:
        return
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def destroy_if_possible(obj: Any) -> None:
    if obj is None:
        return
    destroy = getattr(obj, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass


def collect() -> None:
    gc.collect()
    try:
        from petsc4py import PETSc  # type: ignore

        PETSc.garbage_cleanup()
    except Exception:
        pass
