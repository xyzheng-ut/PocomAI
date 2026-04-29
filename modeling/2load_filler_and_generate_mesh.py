"""
Rebuild colored 3D composites from CSV filler info and export as GLB (or PLY),
now with an RVE box boundary (wireframe or solid).

CSV columns expected:
composite_id,filler_index,type,center_x,center_y,center_z,s1,s2,s3,vec_x,vec_y,vec_z

Type interpretation:
- sphere   : s1=radius,                 s2=0,    s3=0,    vec ignored
- cylinder : s1=radius,  s2=length,     s3=0,    vec=axis
- cube     : s1=s2=s3=side,                      vec = local +Z axis after rotation
- ellipsoid: s1=a,       s2=b (=a),     s3=c,    vec = c-axis direction

Usage examples:
  python reconstruct_with_box.py out_microstructures/csv --out_dir recon --fmt glb
  python reconstruct_with_box.py out_microstructures/csv --box solid --box_alpha 60
  python reconstruct_with_box.py some.csv --box wireframe --edge_radius 0.004
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh

# ---------- Colors (RGBA, 0..255) ----------
COLOR_MAP = {
    "sphere":    [220,  68,  57, 255],  # red-ish
    "cylinder":  [ 46, 134, 222, 255],  # blue-ish
    "cube":      [ 88, 177,  73, 255],  # green-ish
    "ellipsoid": [171,  85, 208, 255],  # purple-ish
}
BOX_WIREFRAME_COLOR = [40, 40, 40, 255]  # dark gray for edges

# ---------- Geometry builders (centered at origin; orientation applied here) ----------
def make_sphere(radius, subdivisions=2):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(radius))

def make_cylinder(radius, length, axis, sections=18):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    axis = np.array([0., 0., 1.]) if n < 1e-12 else axis / n
    m = trimesh.creation.cylinder(radius=float(radius), height=float(length), sections=sections)
    R = trimesh.geometry.align_vectors(np.array([0., 0., 1.]), axis)
    m.apply_transform(R)
    return m

def make_cube(side, z_axis=None):
    m = trimesh.creation.box(extents=[float(side), float(side), float(side)])
    if z_axis is not None:
        z_axis = np.asarray(z_axis, float)
        n = np.linalg.norm(z_axis)
        if n > 1e-12:
            z_axis = z_axis / n
            R = trimesh.geometry.align_vectors(np.array([0., 0., 1.]), z_axis)
            m.apply_transform(R)
    return m

def make_ellipsoid(a, c, c_axis, subdivisions=2):
    base = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    S = np.eye(4)
    S[0,0] = float(a); S[1,1] = float(a); S[2,2] = float(c)
    base.apply_transform(S)
    c_axis = np.asarray(c_axis, float)
    n = np.linalg.norm(c_axis)
    c_axis = np.array([0., 0., 1.]) if n < 1e-12 else c_axis / n
    R = trimesh.geometry.align_vectors(np.array([0., 0., 1.]), c_axis)
    base.apply_transform(R)
    return base

def translate_mesh(mesh, center):
    T = np.eye(4); T[:3, 3] = np.asarray(center, float)
    mesh = mesh.copy(); mesh.apply_transform(T); return mesh

def apply_color(mesh, rgba):
    rgba = np.asarray(rgba, dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return mesh

# ---------- Box boundary (NEW) ----------
def make_box_wireframe(size=1.0, edge_radius=0.003, color=BOX_WIREFRAME_COLOR):
    """
    12 edges as thin cylinders along the unit box [0,size]^3.
    """
    s = float(size)
    # 8 corners
    C = np.array([[0,0,0],[s,0,0],[0,s,0],[s,s,0],[0,0,s],[s,0,s],[0,s,s],[s,s,s]], dtype=float)
    # 12 edges as (start_idx, end_idx)
    edges = [
        (0,1),(2,3),(4,5),(6,7),  # x edges
        (0,2),(1,3),(4,6),(5,7),  # y edges
        (0,4),(1,5),(2,6),(3,7),  # z edges
    ]
    tubes = []
    for a,b in edges:
        p1, p2 = C[a], C[b]
        axis = p2 - p1
        length = np.linalg.norm(axis)
        if length <= 0:  # guard
            continue
        m = make_cylinder(edge_radius, length, axis)
        mid = (p1 + p2) * 0.5
        m = translate_mesh(m, mid)
        tubes.append(apply_color(m, color))
    if len(tubes) == 1:
        return tubes[0]
    return trimesh.util.concatenate(tubes)

def make_box_solid(size=1.0, color=(255,255,255,64)):
    """
    Solid cube with semi-transparent color.
    """
    s = float(size)
    m = trimesh.creation.box(extents=[s,s,s])
    # center at (s/2,s/2,s/2)
    m = translate_mesh(m, [s/2, s/2, s/2])
    return apply_color(m, color)

# ---------- Reconstruction from a single CSV ----------
def reconstruct_from_csv(csv_path, out_dir, fmt="glb",
                         rve_size=1.0, box_mode="wireframe",
                         edge_radius=0.003, box_color=None, box_alpha=255,
                         verbose=True):
    df = pd.read_csv(csv_path)
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    geoms = []

    # --- Add box boundary first (NEW) ---
    if box_mode and box_mode.lower() != "none":
        if box_mode.lower() == "wireframe":
            color = BOX_WIREFRAME_COLOR if box_color is None else list(box_color) + [255]
            box = make_box_wireframe(size=rve_size, edge_radius=edge_radius, color=color)
        elif box_mode.lower() == "solid":
            base = [255, 255, 255] if box_color is None else list(box_color)
            color = base + [int(np.clip(box_alpha, 0, 255))]
            box = make_box_solid(size=rve_size, color=color)
        else:
            raise ValueError("box_mode must be one of: none, wireframe, solid")
        geoms.append(box)

    # --- Rebuild fillers ---
    for _, row in df.iterrows():
        ftype = row["type"]
        cx, cy, cz = float(row["center_x"]), float(row["center_y"]), float(row["center_z"])
        s1, s2, s3 = float(row["s1"]), float(row["s2"]), float(row["s3"])
        vx, vy, vz = float(row["vec_x"]), float(row["vec_y"]), float(row["vec_z"])
        center = np.array([cx, cy, cz], dtype=float)
        vec = np.array([vx, vy, vz], dtype=float)

        if ftype == "sphere":
            mesh = make_sphere(s1)
        elif ftype == "cylinder":
            mesh = make_cylinder(s1, s2, vec)
        elif ftype == "cube":
            mesh = make_cube(s1, z_axis=vec)
        elif ftype == "ellipsoid":
            mesh = make_ellipsoid(s1, s3, vec)  # a=s1, c=s3
        else:
            if verbose:
                print(f"  [warn] Unknown type '{ftype}' in {csv_path.name}; skipping this filler.")
            continue

        mesh = translate_mesh(mesh, center)
        color = COLOR_MAP.get(ftype, [200, 200, 200, 255])
        mesh = apply_color(mesh, color)
        mesh.metadata = {"name": f"{ftype}_{int(row['filler_index'])}"}
        geoms.append(mesh)

    if not geoms:
        if verbose:
            print(f"  [warn] No geometries reconstructed for {csv_path.name}")
        return

    # Build a scene and export
    scene = trimesh.Scene()
    for i, m in enumerate(geoms):
        scene.add_geometry(m, node_name=m.metadata.get("name", f"mesh_{i}") if hasattr(m, "metadata") else f"mesh_{i}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = csv_path.stem.replace("composite_", "recon_") + f".{fmt}"
    out_path = out_dir / out_name

    if fmt.lower() == "glb":
        scene.export(out_path)
    elif fmt.lower() == "ply":
        merged = trimesh.util.concatenate([g for g in geoms if isinstance(g, trimesh.Trimesh)])
        merged.export(out_path)
    else:
        raise ValueError("Unsupported export format. Use 'glb' or 'ply'.")

    if verbose:
        print(f"  [ok] Wrote {out_path}")

# ---------- Batch helper ----------
def gather_csvs(input_path):
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    elif p.is_file() and p.suffix.lower() == ".csv":
        return [p]
    else:
        raise FileNotFoundError("Provide a CSV file or a directory containing CSV files.")

def main():
    ap = argparse.ArgumentParser(description="Reconstruct colored 3D composites from CSV, with box boundary.")
    ap.add_argument("--input", default="/Users/user/Documents/papers/202506diffusion_model/figs/fig1/xct-style/out_microstructures/csv", help="CSV file or directory of CSVs (e.g., out_microstructures/csv)")
    ap.add_argument("--out_dir", default="reconstructed", help="Output directory")
    ap.add_argument("--fmt", default="ply", choices=["glb", "ply"], help="Export format")
    ap.add_argument("--rve_size", type=float, default=1.0, help="Domain size (box spans [0, rve_size]^3)")
    ap.add_argument("--box", dest="box_mode", default="wireframe",
                    choices=["wireframe", "solid", "none"], help="Box boundary style")
    ap.add_argument("--edge_radius", type=float, default=0.003, help="Wireframe edge radius")
    ap.add_argument("--box_color", type=float, nargs=3, metavar=("R","G","B"),
                    help="Override box RGB color (0..255). Example: --box_color 255 200 0")
    ap.add_argument("--box_alpha", type=int, default=64, help="Alpha for solid box (0..255)")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = ap.parse_args()

    csvs = gather_csvs(args.input)
    out_dir = Path(args.out_dir)
    verbose = not args.quiet

    if verbose:
        print(f"Found {len(csvs)} CSV file(s). Exporting {args.fmt.upper()} to: {out_dir.resolve()}")
        print(f"Box: {args.box_mode} | rve_size={args.rve_size}")

    for csv_path in csvs:
        if verbose:
            print(f"- Processing {csv_path.name}")
        reconstruct_from_csv(
            csv_path,
            out_dir,
            fmt=args.fmt,
            rve_size=args.rve_size,
            box_mode=args.box_mode,
            edge_radius=args.edge_radius,
            box_color=[int(c) for c in args.box_color] if args.box_color is not None else None,
            box_alpha=args.box_alpha,
            verbose=verbose
        )

if __name__ == "__main__":
    main()
