"""
Generate polymer-composite microstructures with non-overlapping (touching allowed) fillers:
- Types: sphere, cylinder, cube, ellipsoid (a=b!=c)
- Domain: unit RVE [0,1]^3
- Output per composite: CSV with (type, index, center, sizes, vector)

Columns:
composite_id,filler_index,type,center_x,center_y,center_z,s1,s2,s3,vec_x,vec_y,vec_z
  sphere   : s1=radius,                 s2=0,    s3=0,    vec=(0,0,0)
  cylinder : s1=radius,  s2=length,     s3=0,    vec=axis (unit)
  cube     : s1=side,    s2=side,       s3=side, vec=local +Z axis after rotation
  ellipsoid: s1=a,       s2=b (=a),     s3=c,    vec=c-axis direction (unit)

Dependencies: numpy, pandas, trimesh
"""

import os
import math
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh

# -----------------------------
# Config
# -----------------------------
SEED = 123
RVE_SIZE = 1.0
N_COMPOSITES = 10000                      # how many microstructures to generate
FILLERS_RANGE = (50, 100)               # min/max fillers per composite
TOUCH_TOL = 1e-6                       # allow touching; treat overlap only if strictly inside
SURFACE_SAMPLES = 600                  # points per mesh for fine overlap test
SECTIONS_CYL = 20                      # cylinder tessellation
SUBDIV_SPHERE = 2                      # icosphere subdivisions (2~3 is fine)
EXPORT_STL = False                     # set True to also export STL per composite
OUT_DIR = Path("out_microstructures")
CSV_DIR = OUT_DIR / "csv"
STL_DIR = OUT_DIR / "stl"

# Type mixture probabilities
TYPE_PROBS_list = [{
    "sphere":   1.,
    "cylinder": 0.,
    "cube":     0.,
    "ellipsoid":0.,
},{
    "sphere":   0.,
    "cylinder": 1.,
    "cube":     0.,
    "ellipsoid":0.,
},{
    "sphere":   0.,
    "cylinder": 0.,
    "cube":     1.,
    "ellipsoid":0.,
},{
    "sphere":   0.,
    "cylinder": 0.,
    "cube":     0.,
    "ellipsoid":1.,
},{
    "sphere":   0.5,
    "cylinder": 0.5,
    "cube":     0.,
    "ellipsoid":0.,
},{
    "sphere":   0.5,
    "cylinder": 0.,
    "cube":     0.5,
    "ellipsoid":0.,
},{
    "sphere":   0.5,
    "cylinder": 0.,
    "cube":     0.,
    "ellipsoid":0.5,
},{
    "sphere":   0.,
    "cylinder": 0.5,
    "cube":     0.5,
    "ellipsoid":0.,
},{
    "sphere":   0.,
    "cylinder": 0.5,
    "cube":     0.,
    "ellipsoid":0.5,
},{
    "sphere":   0.,
    "cylinder": 0.,
    "cube":     0.5,
    "ellipsoid":0.5,
},{
    "sphere":   0.333,
    "cylinder": 0.333,
    "cube":     0.333,
    "ellipsoid":0.,
},{
    "sphere":   0.333,
    "cylinder": 0.333,
    "cube":     0.,
    "ellipsoid":0.333,
},{
    "sphere":   0.,
    "cylinder": 0.333,
    "cube":     0.333,
    "ellipsoid":0.333,
},{
    "sphere":   0.333,
    "cylinder": 0.,
    "cube":     0.333,
    "ellipsoid":0.333,
},{
    "sphere":   0.25,
    "cylinder": 0.25,
    "cube":     0.25,
    "ellipsoid":0.25,
}
]

# Size ranges (all in RVE units)

SPHERE_R_RANGE     = (0.04, 0.08)
CYL_RADIUS_RANGE   = (0.03, 0.040)
CYL_LENGTH_RANGE   = (0.10, 0.400)
CUBE_SIDE_RANGE    = (0.070, 0.150)
ELLIPSOID_A_RANGE  = (0.03, 0.050)  # a=b
ELLIPSOID_C_RANGE  = (0.060, 0.150)   # c != a (we'll ensure |c-a| > tiny)

rng = np.random.default_rng(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
if EXPORT_STL:
    STL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Random helpers
# -----------------------------
def rand_unit_vec(rng):
    """Uniform random unit vector on the sphere."""
    phi = rng.uniform(0.0, 2.0*np.pi)
    u = rng.uniform(-1.0, 1.0)
    s = math.sqrt(max(0.0, 1.0 - u*u))
    return np.array([s*math.cos(phi), s*math.sin(phi), u], dtype=float)

def pick_type(rng, TYPE_PROBS):
    types, probs = zip(*TYPE_PROBS.items())
    return rng.choice(types, p=np.array(probs)/np.sum(probs))

def random_center_inside_rve(rng):
    return rng.uniform(0.0, RVE_SIZE, size=3).astype(float)


# -----------------------------
# Mesh builders (centered at origin then transformed)
# -----------------------------
def make_sphere(radius):
    m = trimesh.creation.icosphere(subdivisions=SUBDIV_SPHERE, radius=radius)
    return m

def make_cylinder(radius, length, axis):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    m = trimesh.creation.cylinder(radius=radius, height=length, sections=SECTIONS_CYL)
    R = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), axis)
    m.apply_transform(R)
    return m

def make_cube(side, Rmat=None):
    # Trimesh box extents are (x,y,z)
    m = trimesh.creation.box(extents=[side, side, side])
    if Rmat is not None:
        m.apply_transform(Rmat)
    return m

def make_ellipsoid(a, c, axis):
    """
    Ellipsoid with (a,b,c) where a=b != c; axis is the direction of c-axis.
    Construct from sphere scaled by diag(a,a,c), then align local +Z to axis.
    """
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    base = trimesh.creation.icosphere(subdivisions=SUBDIV_SPHERE, radius=1.0)
    S = np.eye(4)
    S[0,0] = a
    S[1,1] = a
    S[2,2] = c
    base.apply_transform(S)
    R = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), axis)
    base.apply_transform(R)
    return base


# -----------------------------
# Transforms & bounds
# -----------------------------
def translate_mesh(mesh, center):
    T = np.eye(4)
    T[:3, 3] = np.asarray(center, float)
    m = mesh.copy()
    m.apply_transform(T)
    return m

def mesh_fits_in_rve(mesh, eps=0.0):
    lo, hi = mesh.bounds
    return (lo >= -eps).all() and (hi <= (RVE_SIZE + eps)).all()

def bounding_sphere_radius(mesh):
    """
    Cheap conservative radius about mesh.centroid.
    """
    ctr = mesh.centroid
    v = mesh.vertices - ctr
    return float(np.sqrt((v*v).sum(axis=1)).max())


# -----------------------------
# Overlap checks
# -----------------------------
def aabb_maybe_overlap(a, b, allow_touch=True):
    (alo, ahi), (blo, bhi) = a.bounds, b.bounds
    if allow_touch:
        # disjoint only if one max < other min with a strict margin
        return not (
            (ahi[0] < blo[0]) or (ahi[1] < blo[1]) or (ahi[2] < blo[2]) or
            (bhi[0] < alo[0]) or (bhi[1] < alo[1]) or (bhi[2] < alo[2])
        )
    else:
        # touching counts as overlap -> require strict separation to be False
        return not (
            (ahi[0] <= blo[0]) or (ahi[1] <= blo[1]) or (ahi[2] <= blo[2]) or
            (bhi[0] <= alo[0]) or (bhi[1] <= alo[1]) or (bhi[2] <= alo[2])
        )

def approx_mesh_overlap(candidate, existing, samples=SURFACE_SAMPLES, touch_tol=TOUCH_TOL):
    """
    Return True if meshes overlap (penetrate). Touching is allowed.
    We sample surface points and test 'inside' using ray checks.
    """
    # Quick AABB reject
    if not aabb_maybe_overlap(candidate, existing, allow_touch=True):
        return False

    # sample points on both surfaces
    pa = candidate.sample(samples)
    pb = existing.sample(samples)

    # Fast inside tests via ray checks (boundary usually returns False)
    inside_a = existing.contains(pa)
    inside_b = candidate.contains(pb)

    if inside_a.any() or inside_b.any():
        return True  # penetration detected

    # If desired, we can compute approx minimal surface-surface distance and
    # treat <= touch_tol as "touching" (allowed).
    # Using ProximityQuery distance (unsigned)
    qa = trimesh.proximity.ProximityQuery(existing)
    qb = trimesh.proximity.ProximityQuery(candidate)

    # Some backends might raise; guard it
    try:
        da = qa.distance(pa)[0]  # (N,) distances
        db = qb.distance(pb)[0]
        min_d = min(float(np.min(da)), float(np.min(db)))
        # If min_d < -touch_tol we'd call it overlap, but 'distance' is unsigned.
        # The inside() check above already caught penetration. Here just return False (~separate or touching).
        _ = min_d
    except Exception:
        pass

    return False  # no interior points found -> treat as not overlapping (touching OK)

def any_overlap(candidate, accepted_meshes):
    for m in accepted_meshes:
        if approx_mesh_overlap(candidate, m):
            return True
    return False


# -----------------------------
# Size samplers and orientation vectors
# -----------------------------
def sample_sphere_params():
    r = float(rng.uniform(*SPHERE_R_RANGE))
    vec = np.array([0.0, 0.0, 0.0])
    return r, vec

def sample_cylinder_params():
    r = float(rng.uniform(*CYL_RADIUS_RANGE))
    L = float(rng.uniform(*CYL_LENGTH_RANGE))
    axis = rand_unit_vec(rng)
    return r, L, axis

def sample_cube_params():
    s = float(rng.uniform(*CUBE_SIDE_RANGE))
    # random rotation; also record local +Z axis after rotation as a representative vector
    R = trimesh.transformations.random_rotation_matrix()
    z_axis = (R[:3, :3] @ np.array([0.0, 0.0, 1.0])).astype(float)
    return s, R, z_axis

def sample_ellipsoid_params():
    a = float(rng.uniform(*ELLIPSOID_A_RANGE))
    c = float(rng.uniform(*ELLIPSOID_C_RANGE))
    # ensure a!=c visibly
    if abs(a - c) < 1e-3:
        c = a + (0.01 if a < 0.2 else -0.01)
    axis = rand_unit_vec(rng)  # c-axis direction
    return a, c, axis


# -----------------------------
# Per filler placement
# -----------------------------
def build_mesh_and_record(ftype):
    """
    Returns mesh (centered & rotated then translated later), size tuple (s1,s2,s3), vec (unit 3,),
    local_mesh (centered at origin), and a callable that translates to center.
    """
    if ftype == "sphere":
        r, vec = sample_sphere_params()
        m_local = make_sphere(r)
        s = (r, 0.0, 0.0)
        return m_local, s, vec

    if ftype == "cylinder":
        r, L, axis = sample_cylinder_params()
        m_local = make_cylinder(r, L, axis)  # already aligned to axis at origin
        s = (r, L, 0.0)
        return m_local, s, axis

    if ftype == "cube":
        s_side, R, z_axis = sample_cube_params()
        m_local = make_cube(s_side, R)       # rotated at origin
        s = (s_side, s_side, s_side)
        return m_local, s, z_axis

    if ftype == "ellipsoid":
        a, c, axis = sample_ellipsoid_params()
        m_local = make_ellipsoid(a, c, axis) # oriented at origin
        s = (a, a, c)
        return m_local, s, axis

    raise ValueError("Unknown filler type: " + str(ftype))


def place_filler_in_rve(m_local, max_trials=200):
    """
    Randomly translate the given centered mesh into the RVE, ensuring it lies fully inside.
    """
    for _ in range(max_trials):
        center = random_center_inside_rve(rng)
        m = translate_mesh(m_local, center)
        if mesh_fits_in_rve(m, eps=0.0):
            return m, center
    return None, None


# -----------------------------
# Composite generator
# -----------------------------
def generate_composite(composite_id, n_fillers, TYPE_PROBS):
    accepted_meshes = []
    records = []  # rows for CSV
    # For broad-phase pruning: maintain (center, bound_radius)
    centers = []
    radii = []

    attempts = 0
    max_attempts = n_fillers * 500

    while len(records) < n_fillers and attempts < max_attempts:
        attempts += 1
        ftype = pick_type(rng, TYPE_PROBS)
        m_local, size_tuple, vec = build_mesh_and_record(ftype)
        # translate somewhere valid
        m_placed, center = place_filler_in_rve(m_local, max_trials=200)
        if m_placed is None:
            continue

        # Broad-phase: bounding sphere vs existing
        cand_r = bounding_sphere_radius(m_placed)
        ok = True
        for (c0, r0), m0 in zip(zip(centers, radii), accepted_meshes):
            if np.linalg.norm(center - c0) > (cand_r + r0):
                continue  # definitely no overlap
            # precise-ish test
            if approx_mesh_overlap(m_placed, m0):
                ok = False
                break

        if not ok:
            continue

        # Accept
        idx = len(records)
        accepted_meshes.append(m_placed)
        centers.append(center)
        radii.append(cand_r)

        row = {
            "composite_id": composite_id,
            "filler_index": idx,
            "type": ftype,
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "s1": float(size_tuple[0]),
            "s2": float(size_tuple[1]),
            "s3": float(size_tuple[2]),
            "vec_x": float(vec[0]),
            "vec_y": float(vec[1]),
            "vec_z": float(vec[2]),
        }
        records.append(row)

    return accepted_meshes, records


# -----------------------------
# Main
# -----------------------------
def main():
    # all_rows = []
    for j in range(len(TYPE_PROBS_list)):
        for i in range(N_COMPOSITES):
            n_fillers = int(rng.integers(FILLERS_RANGE[0], FILLERS_RANGE[1] + 1))
            meshes, rows = generate_composite(composite_id=i, n_fillers=n_fillers, TYPE_PROBS=TYPE_PROBS_list[j])
            # Write per-composite CSV
            df = pd.DataFrame(rows, columns=[
                "composite_id","filler_index","type",
                "center_x","center_y","center_z","s1","s2","s3",
                "vec_x","vec_y","vec_z"
            ])
            df.to_csv(CSV_DIR / f"composite_{j:02d}{i:04d}.csv", index=False)
            # all_rows.extend(rows)

            # Optional: export STL of the union of fillers (visualization)
            if EXPORT_STL and len(meshes) > 0:
                combined = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
                combined.export(STL_DIR / f"composite_{j:02d}{i:04d}.stl")

        # print(f"[{i+1}/{N_COMPOSITES}] fillers={len(rows)}")

    # Master catalog
    # if all_rows:
    #     pd.DataFrame(all_rows).to_csv(OUT_DIR / "catalog.csv", index=False)

if __name__ == "__main__":
    main()
