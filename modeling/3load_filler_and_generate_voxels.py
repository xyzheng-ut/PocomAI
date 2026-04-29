"""
Voxelize polymer-composite microstructures from CSVs into 64x64x64 labeled grids
and export a colored PLY of the voxel grid.

- Matrix voxels: 0
- Fillers: distinct positive integer labels (default: 1 + filler_index)
- PLY colors: per-filler **index-based** (not by type). Each label gets a unique RGB.

CSV Columns (one row per filler):
composite_id, filler_index, type, center_x, center_y, center_z, s1, s2, s3, vec_x, vec_y, vec_z

Geometry parameters:
- sphere   : s1=radius                                   ; vec ignored
- cylinder : s1=radius, s2=length                        ; vec = axis
- cube     : s1=s2=s3=side                               ; vec = local +Z axis after rotation
- ellipsoid: s1=a, s2=b (=a), s3=c                       ; vec = c-axis (a=b!=c)

Usage:
  python voxelize_with_ply.py <csv_or_dir> \
      --out_dir vox_npy --res 64 --rve_size 1.0 \
      --export_ply --ply_dir vox_ply --ply_mode cubes --max_cubes 300000

Notes:
- `--ply_mode cubes` exports an actual voxel **mesh** (robust in most viewers).
- If the number of non-zero voxels exceeds `--max_cubes`, it automatically falls back to `points`.
- `--ply_mode points` exports a colored **point cloud** at voxel centers (small & fast).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import json
# ---------------------------
# Index-based color palette
# ---------------------------

def hsv_to_rgb(h, s, v):
    """h,s,v in [0,1] -> r,g,b in [0,1]"""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return r, g, b

def label_to_rgb(label):
    """
    Deterministic, well-separated color per integer label (>=1).
    Uses golden-ratio hue spacing + mild lightness jitter for variety.
    """
    phi = 0.6180339887498949
    h = (label * phi) % 1.0
    s = 0.65 + 0.25 * (((label * 37) % 7) / 6.0)      # vary saturation slightly
    v = 0.85 - 0.10 * (((label * 17) % 5) / 4.0)      # vary value slightly
    r, g, b = hsv_to_rgb(h, min(s, 0.95), max(min(v, 0.95), 0.6))
    return (int(r * 255), int(g * 255), int(b * 255))

# ---------------------------
# Rotation / basis utilities
# ---------------------------

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n

def rotation_world_from_local_z(z_axis):
    """
    Build orthonormal basis with z_hat aligned to z_axis.
    Returns R with columns (x_hat, y_hat, z_hat).
    """
    z_hat = _normalize(z_axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_hat = _normalize(np.cross(helper, z_hat))
    y_hat = np.cross(z_hat, x_hat)
    return np.stack([x_hat, y_hat, z_hat], axis=1)

# ---------------------------
# Voxelization helpers
# ---------------------------

def make_grid_centers(res, rve_size=1.0):
    """
    Return 3D arrays X,Y,Z of voxel center coordinates in world space.
    Domain: [0, rve_size]^3, centers at (i+0.5)/res * rve_size.
    """
    g = (np.arange(res, dtype=np.float64) + 0.5) * (rve_size / res)
    X, Y, Z = np.meshgrid(g, g, g, indexing="xy")  # shapes: (res,res,res)
    return X, Y, Z

def mask_sphere(X, Y, Z, center, radius):
    cx, cy, cz = center
    return (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 <= radius * radius + 1e-16

def mask_cylinder(X, Y, Z, center, axis, radius, length):
    cx, cy, cz = center
    u = _normalize(axis)
    dx = X - cx; dy = Y - cy; dz = Z - cz
    t = dx*u[0] + dy*u[1] + dz*u[2]                   # projection
    radial2 = dx*dx + dy*dy + dz*dz - t*t
    inside_caps = np.abs(t) <= (0.5 * length + 1e-12)
    inside_rad  = radial2 <= (radius * radius + 1e-16)
    return inside_caps & inside_rad

def mask_cube(X, Y, Z, center, z_axis, side):
    R = rotation_world_from_local_z(z_axis)
    RT = R.T
    cx, cy, cz = center
    dx = X - cx; dy = Y - cy; dz = Z - cz
    qx = RT[0,0]*dx + RT[0,1]*dy + RT[0,2]*dz
    qy = RT[1,0]*dx + RT[1,1]*dy + RT[1,2]*dz
    qz = RT[2,0]*dx + RT[2,1]*dy + RT[2,2]*dz
    h = 0.5 * side + 1e-12
    return (np.abs(qx) <= h) & (np.abs(qy) <= h) & (np.abs(qz) <= h)

def mask_ellipsoid_ab_c(X, Y, Z, center, c_axis, a, c):
    R = rotation_world_from_local_z(c_axis)
    RT = R.T
    cx, cy, cz = center
    dx = X - cx; dy = Y - cy; dz = Z - cz
    qx = RT[0,0]*dx + RT[0,1]*dy + RT[0,2]*dz
    qy = RT[1,0]*dx + RT[1,1]*dy + RT[1,2]*dz
    qz = RT[2,0]*dx + RT[2,1]*dy + RT[2,2]*dz
    val = (qx*qx + qy*qy) / (a*a + 1e-16) + (qz*qz) / (c*c + 1e-16)
    return val <= 1.0 + 1e-12

# ---------------------------
# PLY export helpers
# ---------------------------

def export_points_ply(vol, X, Y, Z, out_path):
    """
    Export non-zero voxels as a colored point cloud with per-label colors.
    """
    idx = np.where(vol > 0)
    if idx[0].size == 0:
        trimesh.PointCloud(vertices=np.zeros((0, 3)), colors=np.zeros((0, 3), dtype=np.uint8)).export(out_path)
        return

    V = np.stack([X[idx], Y[idx], Z[idx]], axis=1)  # (N,3)
    labels = vol[idx].astype(np.int64)

    # Color by label
    lut = np.zeros((labels.max() + 1, 3), dtype=np.uint8)
    for lab in np.unique(labels):
        if lab <= 0: continue
        lut[lab] = np.array(label_to_rgb(int(lab)), dtype=np.uint8)

    colors = lut[labels]
    trimesh.PointCloud(vertices=V, colors=colors).export(out_path)

def export_cubes_ply(vol, res, rve_size, out_path, max_cubes=int(3e6)):
    """
    Export non-zero voxels as a colored **mesh** of cubes (12 triangles per voxel).
    Automatically falls back to points if too many voxels.
    """
    idx = np.where(vol > 0)
    nvox = int(idx[0].size)
    if nvox == 0:
        # write empty mesh
        trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=np.int64)).export(out_path)
        return
    if nvox > max_cubes:
        # too big: fallback to points
        print(f"  [info] {out_path.name}: {nvox} voxels > max_cubes={max_cubes}; exporting points instead.")
        # build grid centers again for this quick path
        s = rve_size / res
        Xc = (idx[1].astype(np.float64) + 0.5) * s
        Yc = (idx[0].astype(np.float64) + 0.5) * s
        Zc = (idx[2].astype(np.float64) + 0.5) * s
        V = np.stack([Xc, Yc, Zc], axis=1)
        labels = vol[idx].astype(np.int64)
        lut = np.zeros((labels.max() + 1, 3), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab <= 0: continue
            lut[lab] = np.array(label_to_rgb(int(lab)), dtype=np.uint8)
        colors = lut[labels]
        trimesh.PointCloud(vertices=V, colors=colors).export(out_path)
        return

    # Build cubes per label to avoid mixing colors
    s = rve_size / res
    half = 0.5 * s
    # 8 corners of a unit cube centered at origin (scaled by s)
    cube_verts = np.array([
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half],
    ], dtype=np.float64)
    # 12 triangles (two per face), using the above vertex order
    cube_faces = np.array([
        [0,1,2], [0,2,3],   # z-
        [4,5,6], [4,6,7],   # z+
        [0,1,5], [0,5,4],   # y-
        [3,2,6], [3,6,7],   # y+
        [0,3,7], [0,7,4],   # x-
        [1,2,6], [1,6,5],   # x+
    ], dtype=np.int64)

    labels_all = vol[idx].astype(np.int64)
    unique_labels = np.array(sorted(list(set(labels_all.tolist()))), dtype=np.int64)

    verts_all = []
    faces_all = []
    colors_all = []

    v_offset = 0
    for lab in unique_labels:
        mask_lab = labels_all == lab
        iy, ix, iz = idx[0][mask_lab], idx[1][mask_lab], idx[2][mask_lab]
        n = int(iy.size)
        if n == 0:
            continue

        # centers in world coords (note axis mapping)
        centers = np.stack([
            (ix.astype(np.float64) + 0.5) * s,
            (iy.astype(np.float64) + 0.5) * s,
            (iz.astype(np.float64) + 0.5) * s
        ], axis=1)  # (n,3)

        # replicate cube template
        verts = (centers[:, None, :] + cube_verts[None, :, :]).reshape(-1, 3)   # (n*8, 3)

        # faces with offset per cube
        f = (cube_faces[None, :, :] + (np.arange(n)[:, None, None] * 8)).reshape(-1, 3)
        faces_all.append(f + v_offset)

        # per-vertex colors for this label
        rgb = np.array(label_to_rgb(int(lab)), dtype=np.uint8)
        colors = np.tile(rgb[None, :], (n * 8, 1))
        colors_all.append(colors)

        verts_all.append(verts)
        v_offset += n * 8

    if len(verts_all) == 0:
        trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=np.int64)).export(out_path)
        return

    V = np.vstack(verts_all)
    F = np.vstack(faces_all)
    C = np.vstack(colors_all)

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    # attach per-vertex colors (PLY supports vertex colors)
    mesh.visual.vertex_colors = C
    mesh.export(out_path)




# ---------------------------
# Core: voxelize one CSV
# ---------------------------

def voxelize_csv(csv_path, out_dir, res=64, rve_size=1.0, label_from_index=True,
                 export_ply=False, ply_dir=None, ply_mode="cubes", max_cubes=int(3e6),
                 verbose=True):
    df = pd.read_csv(csv_path)
    if df.empty:
        if verbose:
            print(f"[warn] {csv_path.name}: no rows; skipping.")
        return

    df["type"] = df["type"].astype(str).str.strip().str.lower()
    if label_from_index and "filler_index" in df.columns:
        df = df.sort_values("filler_index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    X, Y, Z = make_grid_centers(res, rve_size=rve_size)
    vol = np.zeros((res, res, res), dtype=np.uint16)

    for row_idx, row in df.iterrows():
        ftype = row["type"]
        cx, cy, cz = float(row["center_x"]), float(row["center_y"]), float(row["center_z"])
        s1, s2, s3 = float(row["s1"]), float(row["s2"]), float(row["s3"])
        vx, vy, vz = float(row["vec_x"]), float(row["vec_y"]), float(row["vec_z"])

        label = 1 + int(row["filler_index"]) if (label_from_index and "filler_index" in row) else 1 + row_idx

        if ftype == "sphere":
            mask = mask_sphere(X, Y, Z, (cx, cy, cz), s1)
        elif ftype == "cylinder":
            mask = mask_cylinder(X, Y, Z, (cx, cy, cz), (vx, vy, vz), s1, s2)
        elif ftype == "cube":
            mask = mask_cube(X, Y, Z, (cx, cy, cz), (vx, vy, vz), s1)
        elif ftype == "ellipsoid":
            mask = mask_ellipsoid_ab_c(X, Y, Z, (cx, cy, cz), (vx, vy, vz), s1, s3)
        else:
            if verbose:
                print(f"[warn] {csv_path.name}: unknown type '{ftype}', skipping.")
            continue

        vol[mask & (vol == 0)] = np.uint8(label)  # keep first-claim on touching boundaries

    # Save volume
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_name = csv_path.stem.replace("composite_", "voxel_") + f"_{res}.npz"
    vol_path = out_dir / vol_name
    np.savez_compressed(vol_path, vol=vol)

    # np.save(vol_path, vol)

    # Optional PLY
    if export_ply:
        ply_dir = Path(ply_dir) if ply_dir else out_dir
        ply_dir.mkdir(parents=True, exist_ok=True)
        ply_name = csv_path.stem.replace("composite_", "voxel_") + f"_{res}.ply"
        ply_path = ply_dir / ply_name

        if ply_mode.lower() == "cubes":
            export_cubes_ply(vol, res, rve_size, ply_path, max_cubes=max_cubes)
        elif ply_mode.lower() == "points":
            export_points_ply(vol, X, Y, Z, ply_path)
        else:
            raise ValueError("ply_mode must be 'cubes' or 'points'.")

    if verbose:
        nz = int((vol > 0).sum())
        uniq = np.unique(vol)
        msg = f"[ok] {csv_path.name} -> {vol_name} (nonzero voxels: {nz}, fillers in volume: {uniq.size-1})"
        if export_ply:
            msg += f" + {ply_name} ({ply_mode})"
        print(msg)

# ---------------------------
# Batch driver
# ---------------------------

def gather_csvs(path_like):
    p = Path(path_like)
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    if p.is_file() and p.suffix.lower() == ".csv":
        return [p]
    raise FileNotFoundError("Provide a CSV file or a directory containing CSV files.")

def main():
    ap = argparse.ArgumentParser(description="Voxelize CSVs and export labeled grids + colored PLY.")
    ap.add_argument("--input", default="/home/rc/pythonProject_img23d/ident/csv", help="CSV file or directory (e.g., out_microstructures/csv)")
    ap.add_argument("--out_dir", default="/home/rc/pythonProject_img23d/ident/vox_npy64", help="Directory for .npy volumes")
    ap.add_argument("--res", type=int, default=64, help="Grid resolution per axis")
    ap.add_argument("--rve_size", type=float, default=1.0, help="Physical domain size")
    ap.add_argument("--no_index_labels", action="store_true",
                    help="If set, labels are 1..K by CSV row order (ignore filler_index).")
    ap.add_argument("--export_ply", default=False, action="store_true", help="Export PLY of the voxel grid")
    ap.add_argument("--ply_dir", default="ply_dir", help="Directory for PLY files (defaults to --out_dir)")
    ap.add_argument("--ply_mode", default="cubes", choices=["cubes", "points"],
                    help="PLY as voxel cubes (mesh) or voxel centers (points)")
    ap.add_argument("--max_cubes", type=int, default=int(3e6),
                    help="Max number of cubes before falling back to points")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = ap.parse_args()

    csvs = gather_csvs(args.input)
    if not args.quiet:
        dest = Path(args.out_dir).resolve()
        extra = f", PLY -> {(Path(args.ply_dir).resolve() if args.export_ply and args.ply_dir else dest)}"
        print(f"Found {len(csvs)} CSV file(s). NPY -> {dest}{extra}")

    for csv_path in csvs:
        voxelize_csv(
            csv_path,
            out_dir=args.out_dir,
            res=args.res,
            rve_size=args.rve_size,
            label_from_index=not args.no_index_labels,
            export_ply=args.export_ply,
            ply_dir=args.ply_dir,
            ply_mode=args.ply_mode,
            max_cubes=args.max_cubes,
            verbose=not args.quiet
        )

if __name__ == "__main__":
    main()
