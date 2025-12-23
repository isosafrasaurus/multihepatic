import os
import numpy as np
import nibabel as nib

from dolfin import File, MeshFunction, facets, MPI, BoundaryMesh, SubMesh

from src import (
    Domain1D,
    Domain3D,
    build_mesh_by_counts,
    Simulation,
    Parameters,
    PressureVelocityProblem,
)

from graphnics import TubeFile


def _nifti_best_affine(nii: nib.Nifti1Image) -> np.ndarray:
    
    hdr = nii.header
    aff = nii.affine
    sform, scode = hdr.get_sform(coded=True)
    qform, qcode = hdr.get_qform(coded=True)
    if scode > 0:
        aff = sform
    elif qcode > 0:
        aff = qform
    return np.array(aff, dtype=float)


def _label_world_bounds(nii: nib.Nifti1Image, label_value: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    aff = _nifti_best_affine(nii)
    lab = np.rint(nii.get_fdata(dtype=np.float32)).astype(np.int32)
    mask = (lab == int(label_value))
    idx = np.argwhere(mask)  
    if idx.shape[0] == 0:
        raise ValueError(f"No voxels with label {label_value}")
    world = nib.affines.apply_affine(aff, idx.astype(float))
    return world.min(axis=0), world.max(axis=0), aff


def debug_dump_label_world_coords(
    nii_path: str,
    *,
    label_value: int,
    out_dir: str,
    max_print: int = 50,
    max_csv: int = 200000,
) -> None:
    if MPI.comm_world.rank != 0:
        return

    nii = nib.load(nii_path)
    aff = _nifti_best_affine(nii)
    lab = np.rint(nii.get_fdata(dtype=np.float32)).astype(np.int32)
    mask = (lab == int(label_value))
    idx = np.argwhere(mask)

    print(f"[debug] NIfTI: {nii_path}")
    print(f"[debug] label=={label_value} voxel count: {idx.shape[0]}")
    print(f"[debug] nifti shape (i,j,k): {lab.shape}")
    print(f"[debug] affine used:\n{aff}")

    if idx.shape[0] == 0:
        return

    world = nib.affines.apply_affine(aff, idx.astype(float))
    wmin = world.min(axis=0)
    wmax = world.max(axis=0)
    print(f"[debug] label=={label_value} world bounds min: {wmin}")
    print(f"[debug] label=={label_value} world bounds max: {wmax}")

    print(f"[debug] first {min(max_print, world.shape[0])} label-world coords:")
    for r in world[:max_print]:
        print(f"  {r[0]: .6g}, {r[1]: .6g}, {r[2]: .6g}")

    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"nifti_label{label_value}_world_coords.npy")
    np.save(npy_path, world)
    print(f"[debug] saved all label-world coords to: {npy_path}")

    if world.shape[0] <= max_csv:
        csv_path = os.path.join(out_dir, f"nifti_label{label_value}_world_coords.csv")
        np.savetxt(csv_path, world, delimiter=",", header="x,y,z", comments="")
        print(f"[debug] saved CSV to: {csv_path}")
    else:
        print(f"[debug] CSV skipped (too many points: {world.shape[0]} > {max_csv})")


def _collect_exterior_facets(mesh):
    
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, tdim)  
    ext = []
    for f in facets(mesh):
        if len(f.entities(tdim)) == 1:
            ext.append(f)
    return ext


def _facet_test_points(f, mesh):
    
    pts = [f.midpoint().array().astype(float)]
    try:
        tdim = mesh.topology().dim()
        mesh.init(tdim - 1, 0)  
        v_ids = f.entities(0)
        coords = mesh.coordinates()
        for vid in v_ids:
            pts.append(coords[int(vid)].astype(float))
    except Exception:
        pass
    return pts


def mark_sink_facets_from_nifti_label(
    mesh,
    nii_path: str,
    label_value: int = 3,
    radius_vox: int = 1,
    enforce_distance_gate: bool = True,
) -> MeshFunction:
    nii = nib.load(nii_path)
    aff = _nifti_best_affine(nii)
    inv_aff = np.linalg.inv(aff)

    lab = np.rint(nii.get_fdata(dtype=np.float32)).astype(np.int32)
    if lab.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {lab.shape}")
    mask = (lab == int(label_value))
    if int(mask.sum()) == 0:
        raise ValueError(f"No voxels with label {label_value} found in {nii_path}")

    A = aff[:3, :3]
    voxel_sizes = np.linalg.norm(A, axis=0)  

    
    
    radius_world = float(np.linalg.norm(voxel_sizes * float(radius_vox) + 1e-12))

    if MPI.comm_world.rank == 0:
        print(f"[sink] affine voxel sizes (world units): {voxel_sizes}")
        print(f"[sink] radius_vox={radius_vox} -> radius_world(diag)~{radius_world:g}")

    sink_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    ext = _collect_exterior_facets(mesh)
    if not ext:
        raise RuntimeError("No exterior facets found (unexpected).")

    nx, ny, nz = mask.shape
    n_marked = 0

    for f in ext:
        found = False
        for xm in _facet_test_points(f, mesh):
            
            v = inv_aff @ np.array([xm[0], xm[1], xm[2], 1.0], dtype=float)
            ci, cj, ck = (int(np.round(v[0])), int(np.round(v[1])), int(np.round(v[2])))

            for di in range(-radius_vox, radius_vox + 1):
                ii = ci + di
                if not (0 <= ii < nx):
                    continue
                for dj in range(-radius_vox, radius_vox + 1):
                    jj = cj + dj
                    if not (0 <= jj < ny):
                        continue
                    for dk in range(-radius_vox, radius_vox + 1):
                        kk = ck + dk
                        if not (0 <= kk < nz):
                            continue
                        if not mask[ii, jj, kk]:
                            continue

                        if enforce_distance_gate:
                            vc = aff @ np.array([ii, jj, kk, 1.0], dtype=float)
                            dist = float(np.linalg.norm(vc[:3] - xm))
                            if dist > radius_world:
                                continue

                        found = True
                        break
                    if found:
                        break
                if found:
                    break

            if found:
                break

        if found:
            sink_facets[f] = 1
            n_marked += 1

    if MPI.comm_world.rank == 0:
        print(f"[sink] marked exterior facets: {n_marked}/{len(ext)} ({100.0*n_marked/max(len(ext),1):.2f}%)")

    
    if n_marked == 0 and enforce_distance_gate:
        if MPI.comm_world.rank == 0:
            print("[sink] 0 marked with distance gate; retrying with distance gate OFF (voxel-neighborhood only).")
        return mark_sink_facets_from_nifti_label(
            mesh,
            nii_path,
            label_value=label_value,
            radius_vox=radius_vox,
            enforce_distance_gate=False,
        )

    return sink_facets


def write_boundary_marker_for_paraview(mesh, sink_facets: MeshFunction, out_path: str) -> None:
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, tdim)

    bmesh = BoundaryMesh(mesh, "exterior")
    bmark = MeshFunction("size_t", bmesh, bmesh.topology().dim(), 0)

    parent_map = bmesh.entity_map(tdim - 1)  
    sf = sink_facets.array()
    bm = bmark.array()
    for bc in range(bmesh.num_cells()):
        parent_facet = int(parent_map[bc])
        bm[bc] = int(sf[parent_facet])

    File(out_path) << bmark

    try:
        sink_surface = SubMesh(bmesh, bmark, 1)
        File(out_path.replace(".pvd", "_only_sink.pvd")) << sink_surface
    except Exception as e:
        if MPI.comm_world.rank == 0:
            print(f"[sink] SubMesh boundary extraction failed: {e}")


def main():
    vtk_path = "_data/dataNew.vtk"
    nii_path = "_data/newData.nii"

    out_dir = "_results"
    if MPI.comm_world.rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    debug_dump_label_world_coords(nii_path, label_value=3, out_dir=out_dir, max_print=50)

    Lambda = Domain1D.from_vtk(vtk_path, edge_resolution_exp=1)

    positions = [d["pos"] for _, d in Lambda.graph.nodes(data=True)]
    pos = np.asarray(positions, dtype=float)
    gmin, gmax = pos.min(axis=0), pos.max(axis=0)

    nii = nib.load(nii_path)
    lmin, lmax, _aff = _label_world_bounds(nii, label_value=3)

    lower = np.minimum(gmin, lmin)
    upper = np.maximum(gmax, lmax)

    if MPI.comm_world.rank == 0:
        print(f"[debug] graph bounds min/max: {gmin} / {gmax}")
        print(f"[debug] label3 bounds min/max: {lmin} / {lmax}")
        print(f"[debug] union bounds min/max: {lower} / {upper}")

    
    Omega_mesh, _bounds = build_mesh_by_counts(
        Lambda.graph,
        counts=(16, 16, 16),
        bounds=(lower, upper),
        padding_m=0.0,
        strict_bounds=True,
    )
    Omega = Domain3D(Omega_mesh)

    if MPI.comm_world.rank == 0:
        coords = Omega.mesh.coordinates()
        print(f"[debug] Omega bounds min: {coords.min(axis=0)}")
        print(f"[debug] Omega bounds max: {coords.max(axis=0)}")

    
    Omega_sink_subdomain = mark_sink_facets_from_nifti_label(
        Omega.mesh,
        nii_path,
        label_value=3,
        radius_vox=1,
        enforce_distance_gate=True,
    )

    
    write_boundary_marker_for_paraview(
        Omega.mesh,
        Omega_sink_subdomain,
        os.path.join(out_dir, "omega_boundary_markers.pvd"),
    )
    File(os.path.join(out_dir, "omega_sink_facets_raw.pvd")) << Omega_sink_subdomain

    
    params = Parameters(
        gamma=3.6145827741262347e-05,
        gamma_a=8.225197366649115e-08,
        gamma_R=8.620057937882969e-08,
        mu=1.0e-3,
        k_t=1.0e-10,
        P_in=100.0 * 133.322,
        P_cvp=1.0 * 133.322,
    )

    with Simulation(
        Lambda=Lambda,
        Omega=Omega,
        problem_cls=PressureVelocityProblem,
        Omega_sink_subdomain=Omega_sink_subdomain,
        linear_solver="mumps",
    ) as sim:
        sol = sim.solve(params)

        File(os.path.join(out_dir, "p3d.pvd")) << sol.p3d
        if sol.v3d is not None:
            File(os.path.join(out_dir, "v3d.pvd")) << sol.v3d

        tube_out = TubeFile(Lambda.graph, os.path.join(out_dir, "p1d_tube.pvd"))
        tube_out << sol.p1d

    Lambda.close()
    Omega.close()


if __name__ == "__main__":
    main()
