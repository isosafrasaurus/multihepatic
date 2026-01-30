from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

@dataclass(frozen=True)
class RepairOptions:
    clean_tolerance: float | None = 0.0

    extract_largest_component: bool = True

    use_pymeshfix: bool = True
    pymeshfix_join_components: bool = True
    pymeshfix_remove_smallest_components: bool = True
    pymeshfix_verbose: bool = False

    use_pyvista_fill_holes: bool = False
    pyvista_fill_holes_size: float = 1e9

    compute_consistent_normals: bool = True

@dataclass(frozen=True)
class TetGenOptions:
    order: int = 1

    quality: bool = True
    minratio: float = 1.5

    mindihedral: float = 10.0

    steinerleft: int = 200000

    nobisect: bool = False

    maxvolume: float = -1.0

    verbose: int = 1

    switches: str | None = None

@dataclass(frozen=True)
class OutputOptions:
    write_volume_xdmf: bool = True
    write_boundary_xdmf: bool = True
    write_debug_vtk: bool = True

    fix_negative_tet_volumes: bool = True

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

def report_surface(mesh: "pv.PolyData", label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"Type: {type(mesh)}")
    print(f"Points: {mesh.n_points}")
    print(f"Cells (faces): {mesh.n_cells}")

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

    tets_bad = tets[bad].copy()
    tets_bad[:, [1, 2]] = tets_bad[:, [2, 1]]
    tets[bad] = tets_bad
    return n_bad

def read_surface_vtk(path: Path) -> "pv.PolyData":
    """
    Read a .vtk file and return a PolyData surface.
    If it's not PolyData (e.g., UnstructuredGrid or MultiBlock), extract surface.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input surface mesh not found: {path}")

    data = pv.read(str(path))

    if isinstance(data, pv.MultiBlock):
        data = data.combine(merge_points=True).extract_surface()

    if not isinstance(data, pv.PolyData):
        data = data.extract_surface()

    data = data.triangulate()
    return data

def clean_and_repair_surface(mesh: "pv.PolyData", opts: RepairOptions) -> "pv.PolyData":
    """
    Repair surface mesh to be suitable for tetrahedralization.
    The goal is to end with n_open_edges == 0 (watertight) where possible.
    """
    report_surface(mesh, "Surface (raw)")

    mesh = mesh.clean(tolerance=opts.clean_tolerance) if opts.clean_tolerance is not None else mesh.clean()

    if opts.extract_largest_component:
        mesh = mesh.extract_largest().clean(tolerance=opts.clean_tolerance)

    mesh = mesh.triangulate().clean(tolerance=opts.clean_tolerance)

    if opts.use_pyvista_fill_holes:
        mesh = mesh.fill_holes(opts.pyvista_fill_holes_size).triangulate().clean(tolerance=opts.clean_tolerance)

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

    if opts.compute_consistent_normals:

        auto_orient = bool(getattr(mesh, "n_open_edges", 1) == 0)
        mesh = mesh.compute_normals(
            consistent_normals=True,
            auto_orient_normals=auto_orient,
            inplace=False,
        )

    report_surface(mesh, "Surface (repaired)")

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

    if out_opts.write_debug_vtk:
        try:
            meshio.write(out_dir / f"{base_name}_volume.vtu", vol_mesh, file_format="vtu")
            print(f"Wrote: {out_dir / f'{base_name}_volume.vtu'}")
        except Exception as e:
            print(f"Warning: failed to write VTU debug output: {e}")

def boundary_tri_exists(markers: np.ndarray) -> bool:

    return np.any(markers == -1)

def count_tiny_tets(points: np.ndarray, tets4: np.ndarray) -> tuple[int, float, float, float]:
    vols = np.abs(_tet_signed_volumes(points, tets4))
    v_min = float(vols.min())
    v_med = float(np.median(vols))

    v_tol = max(v_med * 1e-12, 1e-30)

    n_tiny = int(np.count_nonzero(vols < v_tol))
    return n_tiny, v_tol, v_min, v_med

def main() -> None:

    INPUT_SURFACE_VTK = Path(r"data/nii2mesh_liver_sink.vtk")

    OUTPUT_DIR = Path(r"data/liver_sink.xdmf")
    BASENAME = "liver_sink"

    H_MIN = float(os.environ.get("MESH_H_MIN", "0.001"))

# target length-like bound for tet sizing

    repair_opts = RepairOptions(
        clean_tolerance=H_MIN,

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
        maxvolume=-1.0,

        verbose=1,
        switches=None,

    )

    out_opts = OutputOptions(
        write_volume_xdmf=True,
        write_boundary_xdmf=True,
        write_debug_vtk=True,
        fix_negative_tet_volumes=True,
    )

    surface = read_surface_vtk(INPUT_SURFACE_VTK)

    repaired = clean_and_repair_surface(surface, repair_opts)

    if H_MIN > 0:
        repaired = repaired.clean(tolerance=H_MIN, absolute=True).triangulate()
        print(f"Applied minimum resolution: merged points within h_min={H_MIN}")

    if out_opts.write_debug_vtk:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        repaired_path = OUTPUT_DIR / f"{BASENAME}_surface_repaired.vtk"
        try:
            repaired.save(str(repaired_path))
            print(f"Wrote: {repaired_path}")
        except Exception as e:
            print(f"Warning: failed to write repaired surface VTK: {e}")

    tgen = tetgen.TetGen(repaired)
    if tet_opts.switches is not None:
        nodes, tets, attr, triface_markers = tgen.tetrahedralize(switches=tet_opts.switches)
    else:
        H = 0.001

        nodes, tets, attr, triface_markers = tgen.tetrahedralize(
            order=tet_opts.order,

            quality=False,

            minratio=2.0,

            mindihedral=0.0,

            nobisect=True,

            maxvolume_length=H,

            steinerleft=50000,

            verbose=tet_opts.verbose,
        )

        nodes = np.asarray(nodes, dtype=np.float64)

        tets = np.asarray(tets, dtype=np.int32)

    if out_opts.fix_negative_tet_volumes and tets.shape[1] == 4:
        n_fixed = fix_tet_orientations_inplace(nodes, tets)
        if n_fixed:
            print(f"Fixed {n_fixed} inverted tetrahedra by swapping local node ordering.")

        vols = _tet_signed_volumes(nodes, tets[:, :4] if tets.shape[1] >= 4 else tets)
        abs_vols = np.abs(vols)

        v_ref = float(np.median(abs_vols))
        v_max = float(abs_vols.max())
        v_tol = max(v_ref * 1e-12, v_max * 1e-18)

        n_tiny = int(np.count_nonzero(abs_vols <= v_tol))
        if n_tiny:
            v_min = float(abs_vols.min())
            print(f"WARNING: {n_tiny}/{len(abs_vols)} tets have very small volume "
                  f"(min={v_min:.3e}, tol={v_tol:.3e}).")

    tets4 = tets[:, :4]

    n_tiny, v_tol, v_min, v_med = count_tiny_tets(nodes, tets4)
    if n_tiny:
        print(f"Warning: {n_tiny} tets have |V| < {v_tol:.3e} (min={v_min:.3e}, median={v_med:.3e}).")

        if n_tiny / tets4.shape[0] > 1e-4:
            raise RuntimeError("Too many near-degenerate tetrahedra. Adjust repair/TetGen settings.")

    report_volume(nodes, tets[:, :4] if tets.shape[1] >= 4 else tets, "Volume mesh (TetGen)")

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

