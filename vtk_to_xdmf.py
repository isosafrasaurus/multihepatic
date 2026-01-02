"""
surface_vtk_to_tet_xdmf.py

Convert a *surface* mesh (.vtk) into a *tetrahedral volume* mesh suitable for 3D FEM,
and write it as XDMF + HDF5 (i.e., .xdmf + .h5).

NO command-line interface:
- Edit the file paths and configuration in main().

Recommended installs (conda-forge is often easiest for VTK/PyVista):
    pip install pyvista tetgen pymeshfix meshio h5py
or:
    conda install -c conda-forge pyvista tetgen pymeshfix meshio h5py

Key libraries:
- PyVista: read/clean/inspect meshes; provides n_open_edges, clean(), triangulate(), etc.
- PyMeshFix: robust watertight surface repair (holes, self-intersections, degenerates).
- TetGen: tetrahedralize the repaired closed surface.
- meshio: write XDMF/HDF5 for FEM solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np


# ----------------------------
# Config objects
# ----------------------------

@dataclass(frozen=True)
class RepairOptions:
    """
    Repair options for the *surface* mesh.

    The safest default is to use PyMeshFix for watertight repair.
    PyVista's fill_holes() exists but is documented as "known to segfault" in some cases,
    so it's disabled by default.  See PyVista docs. :contentReference[oaicite:2]{index=2}
    """
    # Basic cleanup
    clean_tolerance: float | None = 0.0  # 0.0 means "exact duplicates only"; increase if needed
    extract_largest_component: bool = True

    # Robust watertight repair (recommended)
    use_pymeshfix: bool = True
    pymeshfix_join_components: bool = True
    pymeshfix_remove_smallest_components: bool = True
    pymeshfix_verbose: bool = False

    # Optional (use at own risk; can segfault per docs)
    use_pyvista_fill_holes: bool = False
    pyvista_fill_holes_size: float = 1e9  # in mesh units; "radius of bounding circumsphere" :contentReference[oaicite:3]{index=3}

    # Normals/orientation
    compute_consistent_normals: bool = True  # uses compute_normals() :contentReference[oaicite:4]{index=4}


@dataclass(frozen=True)
class TetGenOptions:
    """
    Options for TetGen tetrahedralization.

    TetGen offers many knobs. Here we expose a few common ones.
    See tetgen.TetGen.tetrahedralize signature for details. :contentReference[oaicite:5]{index=5}
    """
    order: int = 1                 # 1 -> linear tetra (4 nodes), 2 -> quadratic tetra (10 nodes)
    quality: bool = True
    minratio: float = 1.5          # > 1.0 ; closer to 1.0 => higher quality but may add more points :contentReference[oaicite:6]{index=6}
    mindihedral: float = 10.0      # degrees; larger => higher quality :contentReference[oaicite:7]{index=7}
    steinerleft: int = 200000      # allow enough Steiner points; too strict settings can "hang" :contentReference[oaicite:8]{index=8}
    nobisect: bool = False         # if True, reduces Steiner on surface; can help if surface is already good :contentReference[oaicite:9]{index=9}

    # Control element size (optional). If <= 0, no global max volume constraint.
    maxvolume: float = -1.0

    # TetGen verbosity: 0,1,2 per docs :contentReference[oaicite:10]{index=10}
    verbose: int = 1

    # You can also pass raw TetGen switches if you prefer. If not None, overrides many pythonic options.
    switches: str | None = None


@dataclass(frozen=True)
class OutputOptions:
    write_volume_xdmf: bool = True
    write_boundary_xdmf: bool = True
    write_debug_vtk: bool = True        # writes repaired surface and tet mesh in VTK formats for inspection
    fix_negative_tet_volumes: bool = True


# ----------------------------
# Imports (with helpful errors)
# ----------------------------

def _import_or_die(module_name: str, pip_hint: str) -> object:
    try:
        return __import__(module_name)
    except ImportError as e:
        raise ImportError(
            f"Missing dependency '{module_name}'. Install it with:\n  {pip_hint}"
        ) from e


pv = _import_or_die("pyvista", "pip install pyvista  (or: conda install -c conda-forge pyvista)")
tetgen = _import_or_die("tetgen", "pip install tetgen  (or: conda install -c conda-forge tetgen)")
meshio = _import_or_die("meshio", "pip install meshio h5py  (or: conda install -c conda-forge meshio h5py)")

try:
    pymeshfix = __import__("pymeshfix")
except ImportError:
    pymeshfix = None


# ----------------------------
# Utility / reporting
# ----------------------------

def report_surface(mesh: "pv.PolyData", label: str) -> None:
    """Print a quick diagnostic summary for a surface mesh."""
    print(f"\n--- {label} ---")
    print(f"Type: {type(mesh)}")
    print(f"Points: {mesh.n_points}")
    print(f"Cells (faces): {mesh.n_cells}")
    # Open edges are a strong indicator of holes / non-watertight surfaces. :contentReference[oaicite:11]{index=11}
    try:
        print(f"Open edges: {mesh.n_open_edges}")
    except Exception:
        print("Open edges: <unavailable>")
    print(f"Bounds: {mesh.bounds}")


def report_volume(points: np.ndarray, tets: np.ndarray, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"Points: {points.shape[0]}")
    print(f"Tets:   {tets.shape[0]}")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"Bounds: x[{mins[0]}, {maxs[0]}], y[{mins[1]}, {maxs[1]}], z[{mins[2]}, {maxs[2]}]")


def _tet_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """
    Signed volumes for linear tetrahedra.
    Positive means consistent orientation; negative means inverted.
    """
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    # volume = dot(cross(p1-p0, p2-p0), (p3-p0)) / 6
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), (p3 - p0)) / 6.0


def fix_tet_orientations_inplace(points: np.ndarray, tets: np.ndarray) -> int:
    """
    Fix inverted (negative-volume) linear tetrahedra by swapping two vertices.
    Returns how many tets were fixed.
    """
    vols = _tet_signed_volumes(points, tets)
    bad = vols < 0.0
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return 0
    # Swap vertex 1 and 2 for the bad tets
    tets_bad = tets[bad].copy()
    tets_bad[:, [1, 2]] = tets_bad[:, [2, 1]]
    tets[bad] = tets_bad
    return n_bad


# ----------------------------
# Core pipeline
# ----------------------------

def read_surface_vtk(path: Path) -> "pv.PolyData":
    """
    Read a .vtk file and return a PolyData surface.
    If it's not PolyData (e.g., UnstructuredGrid or MultiBlock), extract surface.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input surface mesh not found: {path}")

    data = pv.read(str(path))

    # MultiBlock -> combine
    if isinstance(data, pv.MultiBlock):
        data = data.combine(merge_points=True).extract_surface()

    # If it's not PolyData, extract the outer surface
    if not isinstance(data, pv.PolyData):
        data = data.extract_surface()

    # Ensure triangles
    data = data.triangulate()
    return data


def clean_and_repair_surface(mesh: "pv.PolyData", opts: RepairOptions) -> "pv.PolyData":
    """
    Repair surface mesh to be suitable for tetrahedralization.
    The goal is to end with n_open_edges == 0 (watertight) where possible.
    """
    report_surface(mesh, "Surface (raw)")

    # Clean: merges duplicate points, removes unused points, degenerate cells, etc. :contentReference[oaicite:12]{index=12}
    mesh = mesh.clean(tolerance=opts.clean_tolerance) if opts.clean_tolerance is not None else mesh.clean()

    # Remove small disconnected pieces (optional)
    if opts.extract_largest_component:
        mesh = mesh.extract_largest().clean(tolerance=opts.clean_tolerance)

    # Ensure triangles again
    mesh = mesh.triangulate().clean(tolerance=opts.clean_tolerance)

    # Optional: fill holes with VTK (warning: can segfault per PyVista docs) :contentReference[oaicite:13]{index=13}
    if opts.use_pyvista_fill_holes:
        mesh = mesh.fill_holes(opts.pyvista_fill_holes_size).triangulate().clean(tolerance=opts.clean_tolerance)

    # Robust watertight repair with PyMeshFix (recommended) :contentReference[oaicite:14]{index=14}
    if opts.use_pymeshfix:
        if pymeshfix is None:
            raise ImportError(
                "RepairOptions.use_pymeshfix=True but pymeshfix is not installed.\n"
                "Install it with: pip install pymeshfix  (or conda install -c conda-forge pymeshfix)"
            )
        mfix = pymeshfix.MeshFix(mesh)
        mfix.repair(
            verbose=opts.pymeshfix_verbose,
            joincomp=opts.pymeshfix_join_components,
            remove_smallest_components=opts.pymeshfix_remove_smallest_components,
        )
        mesh = mfix.mesh.triangulate().clean(tolerance=opts.clean_tolerance)

    # Compute normals (optionally). Note: auto_orient_normals assumes closed manifold surface. :contentReference[oaicite:15]{index=15}
    if opts.compute_consistent_normals:
        # Only enable auto-orient if watertight (best-effort safety)
        auto_orient = bool(getattr(mesh, "n_open_edges", 1) == 0)
        mesh = mesh.compute_normals(
            consistent_normals=True,
            auto_orient_normals=auto_orient,
            inplace=False,
        )

    report_surface(mesh, "Surface (repaired)")

    # Final check: watertightness (open edges)
    if hasattr(mesh, "n_open_edges") and mesh.n_open_edges != 0:
        raise RuntimeError(
            f"Surface repair did not produce a watertight mesh. "
            f"Open edges remaining: {mesh.n_open_edges}\n"
            f"Try: increasing pymeshfix_join_components, disabling extract_largest_component if needed,\n"
            f"or pre-repairing in a mesh tool (MeshLab/Blender) before running TetGen."
        )

    return mesh


def tetrahedralize_surface(mesh: "pv.PolyData", opts: TetGenOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Tetrahedralize a watertight surface using TetGen and return:
      nodes (N,3), tets (M,k), attributes (optional), triface_markers

    See tetgen.TetGen.tetrahedralize options. :contentReference[oaicite:16]{index=16}
    """
    # Construct TetGen object
    tgen = tetgen.TetGen(mesh)

    if opts.switches is not None:
        nodes, tets, attr, triface_markers = tgen.tetrahedralize(switches=opts.switches)
    else:
        nodes, tets, attr, triface_markers = tgen.tetrahedralize(
            order=opts.order,
            quality=opts.quality,
            minratio=opts.minratio,
            mindihedral=opts.mindihedral,
            steinerleft=opts.steinerleft,
            nobisect=opts.nobisect,
            maxvolume=opts.maxvolume,
            verbose=opts.verbose,
        )

    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)

    return nodes, tets, attr, triface_markers


def write_xdmf_with_h5(mesh: "meshio.Mesh", xdmf_path: Path) -> None:
    """
    Write XDMF with HDF5 heavy data using meshio.
    We force writing from the output directory to keep the .h5 beside the .xdmf.
    """
    xdmf_path = xdmf_path.with_suffix(".xdmf")
    xdmf_path.parent.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    try:
        os.chdir(str(xdmf_path.parent))
        meshio.write(xdmf_path.name, mesh, file_format="xdmf")
    finally:
        os.chdir(cwd)

    print(f"Wrote: {xdmf_path}")
    print(f"(meshio will also create the companion HDF5 file next to it)")


def export_volume_and_boundary(
    nodes: np.ndarray,
    tets: np.ndarray,
    tgen_obj: "tetgen.TetGen | None",
    out_dir: Path,
    base_name: str,
    out_opts: OutputOptions,
    attr: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine cell type
    if tets.shape[1] == 4:
        tet_cell_type = "tetra"
    elif tets.shape[1] == 10:
        tet_cell_type = "tetra10"
    else:
        raise ValueError(f"Unexpected tet connectivity shape {tets.shape}. Expected 4 or 10 nodes per tet.")

    cell_data = {}
    if attr is not None:
        attr = np.asarray(attr).reshape(-1)
        cell_data["region_id"] = [attr]

    vol_mesh = meshio.Mesh(
        points=nodes,
        cells=[(tet_cell_type, tets)],
        cell_data=cell_data if cell_data else None,
    )

    if out_opts.write_volume_xdmf:
        write_xdmf_with_h5(vol_mesh, out_dir / f"{base_name}_volume.xdmf")

    # Optional: boundary surface (triangles) for BCs.
    # TetGen provides triangle face markers; exterior faces are typically marked -1. :contentReference[oaicite:17]{index=17}
    if out_opts.write_boundary_xdmf:
        if tgen_obj is None:
            print("Boundary export skipped (no TetGen object provided).")
        else:
            tri = np.asarray(tgen_obj.trifaces, dtype=np.int64)
            markers = np.asarray(tgen_obj.triface_markers).reshape(-1)
            boundary_tri = tri[markers == -1] if boundary_tri_exists(markers) else tri

            surf_mesh = meshio.Mesh(
                points=nodes,
                cells=[("triangle", boundary_tri)],
            )
            write_xdmf_with_h5(surf_mesh, out_dir / f"{base_name}_boundary.xdmf")

    # Debug formats (easy to inspect in ParaView)
    if out_opts.write_debug_vtk:
        try:
            meshio.write(out_dir / f"{base_name}_volume.vtu", vol_mesh, file_format="vtu")
            print(f"Wrote: {out_dir / f'{base_name}_volume.vtu'}")
        except Exception as e:
            print(f"Warning: failed to write VTU debug output: {e}")


def boundary_tri_exists(markers: np.ndarray) -> bool:
    # Some builds may use different marker conventions; -1 is documented as exterior in tetgen docs. :contentReference[oaicite:18]{index=18}
    return np.any(markers == -1)

def count_tiny_tets(points: np.ndarray, tets4: np.ndarray) -> tuple[int, float, float, float]:
    vols = np.abs(_tet_signed_volumes(points, tets4))
    v_min = float(vols.min())
    v_med = float(np.median(vols))

    # scale-aware tolerance: "tiny compared to typical element"
    # (tune 1e-12 -> 1e-10 if you want stricter rejection)
    v_tol = max(v_med * 1e-12, 1e-30)

    n_tiny = int(np.count_nonzero(vols < v_tol))
    return n_tiny, v_tol, v_min, v_med


# ----------------------------
# Main (hard-coded paths & config)
# ----------------------------

def main() -> None:
    # ---- Hard-coded file paths (EDIT THESE) ----
    INPUT_SURFACE_VTK = Path(r"data/nii2mesh_liver_sink.vtk")

    OUTPUT_DIR = Path(r"data/liver_sink.xdmf")
    BASENAME = "liver_sink"

    # ---- Size knobs ----
    H_MIN = float(os.environ.get("MESH_H_MIN", "0.001"))  # minimum geometric resolution
    # Optional: global coarsening knob (uncomment if you want fewer tets)
    # H_MAX = float(os.environ.get("MESH_H_MAX", "-1.0"))  # target length-like bound for tet sizing

    # ---- Options (EDIT THESE) ----
    repair_opts = RepairOptions(
        clean_tolerance=H_MIN,  # <-- was 0.0
        extract_largest_component=True,
        use_pymeshfix=True,
        pymeshfix_join_components=True,
        pymeshfix_remove_smallest_components=True,
        pymeshfix_verbose=False,
        use_pyvista_fill_holes=False,
        pyvista_fill_holes_size=1e9,
        compute_consistent_normals=True,
    )

    tet_opts = TetGenOptions(
        order=1,
        quality=True,
        minratio=1.5,
        mindihedral=10.0,
        steinerleft=300000,
        nobisect=False,
        maxvolume=-1.0,   # set >0 to control global element size
        verbose=1,
        switches=None,    # or e.g. "pq1.2/10Y" (advanced)
    )

    out_opts = OutputOptions(
        write_volume_xdmf=True,
        write_boundary_xdmf=True,
        write_debug_vtk=True,
        fix_negative_tet_volumes=True,
    )

    # ---- Pipeline ----
    surface = read_surface_vtk(INPUT_SURFACE_VTK)

    repaired = clean_and_repair_surface(surface, repair_opts)
    # Enforce minimum geometric resolution (collapse tiny edges/triangles)
    if H_MIN > 0:
        repaired = repaired.clean(tolerance=H_MIN, absolute=True).triangulate()
        print(f"Applied minimum resolution: merged points within h_min={H_MIN}")

    # Save repaired surface for inspection (optional)
    if out_opts.write_debug_vtk:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        repaired_path = OUTPUT_DIR / f"{BASENAME}_surface_repaired.vtk"
        try:
            repaired.save(str(repaired_path))
            print(f"Wrote: {repaired_path}")
        except Exception as e:
            print(f"Warning: failed to write repaired surface VTK: {e}")

    # Tetrahedralize
    # We keep a TetGen object around to export boundary faces reliably.
    tgen = tetgen.TetGen(repaired)
    if tet_opts.switches is not None:
        nodes, tets, attr, triface_markers = tgen.tetrahedralize(switches=tet_opts.switches)
    else:
        H = 0.001  # <-- your h resolution (units = mesh units)

        nodes, tets, attr, triface_markers = tgen.tetrahedralize(
            order=tet_opts.order,

            # If you're OOM'ing, start with quality=False and relax later
            quality=False,  # faster + fewer refinements :contentReference[oaicite:5]{index=5}
            minratio=2.0,  # more permissive than 1.5
            mindihedral=0.0,  # more permissive than 10

            # Strongly consider preserving the surface if it's already detailed
            nobisect=True,  # maps to TetGen -Y :contentReference[oaicite:6]{index=6}

            # Cap the size using an h-like control:
            maxvolume_length=H,  # <-- key change :contentReference[oaicite:7]{index=7}

            # Keep this bounded; huge values can balloon memory
            steinerleft=50000,  # start smaller; raise only if needed :contentReference[oaicite:8]{index=8}

            verbose=tet_opts.verbose,
        )

        nodes = np.asarray(nodes, dtype=np.float64)

        # IMPORTANT: keep connectivity compact (tetgen returns int32) :contentReference[oaicite:9]{index=9}
        tets = np.asarray(tets, dtype=np.int32)

    # Fix inverted tets (linear only)
    if out_opts.fix_negative_tet_volumes and tets.shape[1] == 4:
        n_fixed = fix_tet_orientations_inplace(nodes, tets)
        if n_fixed:
            print(f"Fixed {n_fixed} inverted tetrahedra by swapping local node ordering.")

        vols = _tet_signed_volumes(nodes, tets[:, :4] if tets.shape[1] >= 4 else tets)
        abs_vols = np.abs(vols)

        # Scale-aware "near zero" threshold:
        #   - v_ref uses a robust scale (median)
        #   - v_tol is extremely small relative to the mesh
        v_ref = float(np.median(abs_vols))
        v_max = float(abs_vols.max())
        v_tol = max(v_ref * 1e-12, v_max * 1e-18)

        n_tiny = int(np.count_nonzero(abs_vols <= v_tol))
        if n_tiny:
            v_min = float(abs_vols.min())
            print(f"WARNING: {n_tiny}/{len(abs_vols)} tets have very small volume "
                  f"(min={v_min:.3e}, tol={v_tol:.3e}).")

            # If you prefer to DROP them instead of keeping them:
            # keep = abs_vols > v_tol
            # tets = tets[keep]
            # print(f"Dropped {n_tiny} tiny-volume tets; remaining {tets.shape[0]} tets.")

    tets4 = tets[:, :4]  # linear tets only
    n_tiny, v_tol, v_min, v_med = count_tiny_tets(nodes, tets4)
    if n_tiny:
        print(f"Warning: {n_tiny} tets have |V| < {v_tol:.3e} (min={v_min:.3e}, median={v_med:.3e}).")
        # Only hard-fail if it’s more than, say, 0.01% of the mesh
        if n_tiny / tets4.shape[0] > 1e-4:
            raise RuntimeError("Too many near-degenerate tetrahedra. Adjust repair/TetGen settings.")

    report_volume(nodes, tets[:, :4] if tets.shape[1] >= 4 else tets, "Volume mesh (TetGen)")

    # Export XDMF/HDF5 (+ boundary mesh)
    export_volume_and_boundary(
        nodes=nodes,
        tets=tets,
        tgen_obj=tgen,
        out_dir=OUTPUT_DIR,
        base_name=BASENAME,
        out_opts=out_opts,
        attr=attr,
    )


if __name__ == "__main__":
    main()
