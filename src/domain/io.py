from __future__ import annotations

from pathlib import Path
import numpy as np


def _is_section_header(line: str) -> bool:
    """Return True if a line likely starts a new VTK legacy section."""
    s = line.strip()
    if not s:
        return False
    # Common legacy POLYDATA section headers
    return s.startswith(
        (
            "POINTS",
            "LINES",
            "POLYGONS",
            "VERTICES",
            "TRIANGLE_STRIPS",
            "POINT_DATA",
            "CELL_DATA",
            "FIELD",
            "SCALARS",
            "LOOKUP_TABLE",
        )
    )


def _read_n_tokens(lines: list[str], i0: int, n: int) -> tuple[list[str], int]:
    """Read at least n whitespace-separated tokens starting from lines[i0]."""
    toks: list[str] = []
    i = i0
    while len(toks) < n and i < len(lines):
        s = lines[i].strip()
        if s:
            toks.extend(s.split())
        i += 1
    return toks, i


def read_vtk_legacy_polydata_ascii(path: str | Path) -> tuple[
    np.ndarray, list[np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """
    Minimal reader for *ASCII legacy* VTK POLYDATA files containing:
      - POINTS
      - LINES (either classic legacy format or the OFFSETS/CONNECTIVITY style)
      - POINT_DATA and/or CELL_DATA with either FIELD or SCALARS arrays

    Returns:
      points: (npoints, 3) float64
      lines:  list of 1D int arrays (each array is a polyline's point indices)
      point_data: dict[name] -> ndarray
      cell_data:  dict[name] -> ndarray
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    lines_txt = p.read_text().splitlines()
    nlines_txt = len(lines_txt)

    points: np.ndarray | None = None
    polylines: list[np.ndarray] = []
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    i = 0
    while i < nlines_txt:
        line = lines_txt[i].strip()
        if not line:
            i += 1
            continue

        # ---- POINTS ----
        if line.startswith("POINTS"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS line: {line!r}")
            npts = int(parts[1])
            # dtype token in parts[2] is ignored (float/double)
            i += 1
            toks, i = _read_n_tokens(lines_txt, i, 3 * npts)
            if len(toks) < 3 * npts:
                raise ValueError(f"POINTS section ended early (need {3*npts} floats).")
            arr = np.asarray(toks[: 3 * npts], dtype=np.float64)
            points = arr.reshape((npts, 3))
            continue

        # ---- LINES ----
        if line.startswith("LINES"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed LINES line: {line!r}")
            n_cells = int(parts[1])
            total_ints = int(parts[2])
            i += 1
            # Skip blank lines
            while i < nlines_txt and not lines_txt[i].strip():
                i += 1
            if i >= nlines_txt:
                raise ValueError("LINES section ended unexpectedly.")

            nxt = lines_txt[i].strip()
            # VTK 5.1 legacy "OFFSETS ... / CONNECTIVITY ..." style
            if nxt.startswith("OFFSETS"):
                # OFFSETS <dtype>
                i += 1
                offsets_toks: list[str] = []
                while i < nlines_txt:
                    s = lines_txt[i].strip()
                    if not s:
                        i += 1
                        continue
                    if s.startswith("CONNECTIVITY"):
                        break
                    offsets_toks.extend(s.split())
                    i += 1
                if i >= nlines_txt or not lines_txt[i].strip().startswith("CONNECTIVITY"):
                    raise ValueError("LINES section missing CONNECTIVITY header.")
                # CONNECTIVITY <dtype>
                i += 1
                conn_toks: list[str] = []
                while i < nlines_txt:
                    s = lines_txt[i].strip()
                    if not s:
                        i += 1
                        continue
                    if _is_section_header(s) and not s[0].isdigit() and not s[0] == "-":
                        # Next section
                        break
                    # Heuristic: stop when we hit a known section header (POINT_DATA/CELL_DATA/etc.)
                    if s.startswith(("POINT_DATA", "CELL_DATA", "POLYGONS", "VERTICES", "TRIANGLE_STRIPS", "LINES", "POINTS")):
                        break
                    conn_toks.extend(s.split())
                    i += 1

                offsets = np.asarray(offsets_toks, dtype=np.int64)
                conn = np.asarray(conn_toks, dtype=np.int64)

                # OFFSETS is typically length n_cells+1; be forgiving if last is omitted
                if offsets.size == n_cells:
                    offsets = np.concatenate([offsets, [conn.size]])
                if offsets.size < 2:
                    raise ValueError("OFFSETS array too short.")

                # Trust offsets rather than the header count if they disagree
                n_cells_actual = int(offsets.size - 1)
                if n_cells_actual != n_cells:
                    n_cells = n_cells_actual

                if offsets[0] != 0:
                    offsets = offsets - offsets[0]
                if offsets[-1] > conn.size:
                    raise ValueError("OFFSETS indicates more connectivity entries than present.")

                for c in range(n_cells):
                    a = int(offsets[c])
                    b = int(offsets[c + 1])
                    pts_idx = conn[a:b]
                    if pts_idx.size >= 2:
                        polylines.append(pts_idx.astype(np.int64, copy=False))
                continue

            # Classic legacy LINES format:
            #   total_ints integers follow, grouped as:
            #     k i0 i1 ... i{k-1}
            # repeated n_cells times.
            toks, i = _read_n_tokens(lines_txt, i, total_ints)
            if len(toks) < total_ints:
                raise ValueError(f"LINES section ended early (need {total_ints} ints).")
            ints = list(map(int, toks[:total_ints]))
            pos = 0
            for _ in range(n_cells):
                if pos >= len(ints):
                    break
                k = int(ints[pos]); pos += 1
                pts_idx = np.asarray(ints[pos:pos + k], dtype=np.int64)
                pos += k
                if pts_idx.size >= 2:
                    polylines.append(pts_idx)
            continue

        # ---- POINT_DATA / CELL_DATA ----
        if line.startswith("POINT_DATA") or line.startswith("CELL_DATA"):
            is_point = line.startswith("POINT_DATA")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed {parts[0]} line: {line!r}")
            n = int(parts[1])
            i += 1
            # Skip blanks
            while i < nlines_txt and not lines_txt[i].strip():
                i += 1
            if i >= nlines_txt:
                break

            def store(name: str, arr: np.ndarray) -> None:
                (point_data if is_point else cell_data)[name] = arr

            # FIELD style
            if lines_txt[i].strip().startswith("FIELD"):
                parts2 = lines_txt[i].split()
                if len(parts2) < 3:
                    raise ValueError(f"Malformed FIELD line: {lines_txt[i]!r}")
                n_arrays = int(parts2[2])
                i += 1
                for _ in range(n_arrays):
                    header = lines_txt[i].split()
                    if len(header) < 4:
                        raise ValueError(f"Malformed FIELD array header: {lines_txt[i]!r}")
                    name = header[0]
                    ncomp = int(header[1])
                    ntuple = int(header[2])
                    # dtype header[3] ignored (float/double/int)
                    i += 1
                    nvals = ncomp * ntuple
                    toks_vals, i = _read_n_tokens(lines_txt, i, nvals)
                    if len(toks_vals) < nvals:
                        raise ValueError(f"FIELD array {name!r} ended early.")
                    vals = np.asarray(toks_vals[:nvals], dtype=np.float64)
                    if ncomp == 1:
                        store(name, vals.reshape((ntuple,)))
                    else:
                        store(name, vals.reshape((ntuple, ncomp)))
                continue

            # SCALARS style
            if lines_txt[i].strip().startswith("SCALARS"):
                # SCALARS name type [numComp]
                scal = lines_txt[i].split()
                if len(scal) < 3:
                    raise ValueError(f"Malformed SCALARS line: {lines_txt[i]!r}")
                name = scal[1]
                ncomp = int(scal[3]) if len(scal) >= 4 else 1
                i += 1
                # Optional LOOKUP_TABLE line
                if i < nlines_txt and lines_txt[i].strip().startswith("LOOKUP_TABLE"):
                    i += 1
                nvals = n * ncomp
                toks_vals, i = _read_n_tokens(lines_txt, i, nvals)
                if len(toks_vals) < nvals:
                    raise ValueError(f"SCALARS array {name!r} ended early.")
                vals = np.asarray(toks_vals[:nvals], dtype=np.float64)
                if ncomp == 1:
                    store(name, vals.reshape((n,)))
                else:
                    store(name, vals.reshape((n, ncomp)))
                continue

            # If we reach here, we don't support the next data layout; skip.
            continue

        i += 1

    if points is None:
        raise ValueError(f"{path!r} does not contain a POINTS section.")
    if len(polylines) == 0:
        raise ValueError(f"{path!r} does not contain any LINES connectivity.")
    return points, polylines, point_data, cell_data