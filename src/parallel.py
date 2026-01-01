from __future__ import annotations

import faulthandler
import os
import socket
import sys
import time
import traceback
from typing import Callable, Dict, List

from mpi4py import MPI


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _envpick(keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ[k]
    return out


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

    # Optional stack dumps if hung
    if os.environ.get("DEBUG_FAULTHANDLER", "1") == "1":
        try:
            faulthandler.enable()
            sec = float(os.environ.get("DEBUG_DUMP_EVERY", "30"))
            faulthandler.dump_traceback_later(sec, repeat=True, file=sys.stderr)
        except Exception:
            pass


def rank_print(comm: MPI.Comm) -> Callable[[str], None]:
    rank = comm.rank
    size = comm.size
    host = socket.gethostname()
    pid = os.getpid()

    def rprint(msg: str) -> None:
        print(f"[{_now()}] [rank {rank}/{size}] [pid {pid}] [{host}] {msg}", flush=True)

    return rprint


def barrier(comm: MPI.Comm, tag: str, rprint: Callable[[str], None] | None = None) -> None:
    if os.environ.get("DEBUG_BARRIERS", "0") != "1":
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
