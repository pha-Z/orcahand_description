"""
Orca hand Range of Motion (ROM) point cloud test.

Samples random joint configurations within limits, runs forward kinematics,
and records 3D positions of key bodies (palm + fingertips) to build a
point cloud of the hand's reachable workspace.

Usage (from repo root):
  python orcahand_description/rom_test/run_rom_test.py [--samples N] [--output DIR] [--no-ply]

Requires: orcahand_description mesh assets (STL files) next to the MJCF.
"""

import argparse
import os
import sys

import numpy as np
import mujoco

# Bodies whose positions we record for the ROM point cloud (fingertips + palm)
ROM_BODY_NAMES = [
    # "right_palm",
    "right_index_ip",
    "right_middle_ip",
    "right_ring_ip",
    "right_pinky_ip",
    "right_thumb_dp",
]

# Optional: offset from body origin to "tip" point, in body local frame (meters).
# Use this to track a point on the finger tip instead of the body frame origin.
# Keys must match ROM_BODY_NAMES; missing bodies use [0,0,0].
# Example: +Z along the distal phalanx → [0, 0, 0.02] (2 cm).
ROM_BODY_TIP_OFFSETS = {
    # "right_palm": np.array([0.0, 0.0, 0.0]),
    "right_index_ip": np.array([-0.00584213, -0.00020146, 0.04352012]),
    "right_middle_ip": np.array([-0.00584213, -0.00020146, 0.04352012]),
    "right_ring_ip": np.array([-0.00584213, -0.00020146, 0.04352012]),
    "right_pinky_ip": np.array([-0.00584213, -0.00020146, 0.04352012]),
    "right_thumb_dp": np.array([ 0.0, 0.0, 0.031032784] ),
}


def get_repo_root():
    """
    Return the workspace root (parent of `orcahand_description/`).

    `rom_test/` lives inside `orcahand_description/`, so we resolve paths from
    this script location rather than assuming a standalone repo.
    """
    rom_test_dir = os.path.dirname(os.path.abspath(__file__))
    orcahand_description_dir = os.path.dirname(rom_test_dir)
    return os.path.dirname(orcahand_description_dir)


def get_rom_test_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_orcahand_description_dir():
    return os.path.dirname(get_rom_test_dir())


def get_model_path(repo_root=None):
    """Path to scene XML that includes Orca hand + default classes (bone, skin, collision)."""
    _ = repo_root  # kept for backward-compat with older calls
    return os.path.join(get_orcahand_description_dir(), "scene_right_extended.xml")


def load_model(model_path):
    """Load MuJoCo model from scene XML (includes hand + default geom classes)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model/scene XML not found: {model_path}")
    return mujoco.MjModel.from_xml_path(model_path)


def get_joint_limits_and_qpos_indices(model):
    """
    Return list of (joint_name, qpos_index, low, high) for all joints that have
    position limits (revolute/prismatic). Skips fixed joints.
    """
    result = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is None:
            print('no name found for njnt index i=', i, '. skipping.')
            continue
        
        # if name == "right_wrist":
        #     print("temporarily fixing joint 'right_wrist' (not sampling it).")
        #     continue
        
        lo, hi = model.jnt_range[i, 0], model.jnt_range[i, 1]
        if lo >= hi:
            print('lo >= hi for njnt index i=', i, '(name=', name, '). skipping.')
            continue
        qposadr = model.jnt_qposadr[i]
        result.append((name, qposadr, lo, hi))
    return result


def get_body_ids_and_names(model, body_names):
    """Return (list of body IDs, list of names found) in same order; skip missing."""
    ids = []
    found = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(bid)
            found.append(name)
        else:
            sys.stderr.write(f"Warning: body '{name}' not found, skipping.\n")
    return ids, found


def sample_qpos(model, data, joint_specs, rng):
    """Sample one random qpos within joint limits and write into data.qpos."""
    data.qpos[:] = model.qpos0
    for _name, qposadr, lo, hi in joint_specs:
        data.qpos[qposadr] = rng.uniform(lo, hi)


def _mesh_vertices_world(model, data, geom_id):
    """Return (N, 3) world positions of mesh vertices for this geom, or empty array if not a mesh."""
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        return np.zeros((0, 3))
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return np.zeros((0, 3))
    start = int(model.mesh_vertadr[mesh_id])
    nv = int(model.mesh_vertnum[mesh_id])
    mv = np.array(model.mesh_vert)
    if mv.ndim == 2:
        verts = mv[start : start + nv].copy()
    else:
        verts = mv[3 * start : 3 * (start + nv)].reshape(-1, 3)
    R = np.array(data.geom_xmat[geom_id]).reshape(3, 3)
    pos = data.geom_xpos[geom_id]
    return (verts @ R.T) + pos


def compute_tip_from_mesh(model, data, body_id, direction="body_z"):
    """
    Compute a single "tip" point for this body from its mesh geometry.
    direction: "body_z" = furthest point along body's +Z axis;
               "outward" = furthest from body origin (any direction).
    Returns world position (3,) or body xpos if no mesh found.
    """
    body_pos = data.xpos[body_id].copy()
    R = np.array(data.xmat[body_id]).reshape(3, 3)
    body_z = R[:, 2]

    best_pt = body_pos.copy()
    best_score = -np.inf

    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        verts = _mesh_vertices_world(model, data, g)
        if len(verts) == 0:
            continue
        if direction == "body_z":
            # Furthest along body +Z from body origin
            scores = (verts - body_pos) @ body_z
        else:
            # Furthest distance from body origin
            scores = np.linalg.norm(verts - body_pos, axis=1)
        imax = np.argmax(scores)
        if scores[imax] > best_score:
            best_score = scores[imax]
            best_pt = verts[imax]

    return best_pt


def collect_body_positions(data, body_ids, tip_offsets=None, model=None, use_mesh_tips=False):
    """
    Return (len(body_ids), 3) array of world positions.
    If tip_offsets is given (list of (3,) arrays, same length as body_ids),
    each position is body origin + body rotation @ offset (so we track a point
    on the tip instead of the body frame origin).
    If use_mesh_tips is True and model is given, each position is the mesh-derived
    tip (furthest point along body +Z) instead of offset-based.
    """
    out = np.zeros((len(body_ids), 3))
    for i, bid in enumerate(body_ids):
        if use_mesh_tips and model is not None:
            out[i] = compute_tip_from_mesh(model, data, bid, direction="body_z")
        else:
            out[i] = data.xpos[bid].copy()
            if tip_offsets is not None and i < len(tip_offsets):
                off = tip_offsets[i]
                if np.any(off != 0):
                    R = np.array(data.xmat[bid]).reshape(3, 3)
                    out[i] += R @ off
    return out


def run_rom_sampling(model, data, joint_specs, body_ids, n_samples, rng, tip_offsets=None, use_mesh_tips=False):
    """
    Sample n_samples joint configs, run FK, return point cloud (N, 3) and
    optional per-point labels (body index) for downstream use.
    """
    n_bodies = len(body_ids)
    points = []
    labels = []

    for _ in range(n_samples):
        sample_qpos(model, data, joint_specs, rng)
        mujoco.mj_forward(model, data)
        pos = collect_body_positions(data, body_ids, tip_offsets, model=model, use_mesh_tips=use_mesh_tips)
        points.append(pos)
        labels.append(np.arange(n_bodies, dtype=np.int32))

    points = np.concatenate(points, axis=0)
    labels = np.concatenate(labels, axis=0)
    return points, labels


def save_pointcloud_npy(path, points):
    np.save(path, points.astype(np.float32))


def save_pointcloud_ply(path, points, labels=None, color_scheme="blue_pink"):
    try:
        import open3d as o3d
    except ImportError:
        sys.stderr.write("open3d not installed; skipping .ply export.\n")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color by body index for visualization; different hue ranges per scheme.
    if labels is not None and len(labels) == len(points):
        n_bodies = int(labels.max()) + 1
        colors = np.zeros((len(points), 3))

        for i in range(n_bodies):
            mask = labels == i
            if not np.any(mask):
                continue

            t = i / max(1, n_bodies - 1)  # 0 .. 1 along the range

            if color_scheme in {"blue_pink", "auto"}:
                # Interpolate from blue to pink.
                c0 = np.array([0.2, 0.2, 1.0])   # blue-ish
                c1 = np.array([1.0, 0.0, 1.0])   # pink/magenta
            elif color_scheme == "green_yellow":
                # Interpolate from green to yellow.
                c0 = np.array([0.0, 0.8, 0.0])   # green
                c1 = np.array([1.0, 1.0, 0.0])   # yellow
            else:
                c0 = c1 = np.array([1.0, 0.0, 1.0])  # fallback pink

            rgb = (1.0 - t) * c0 + t * c1
            colors[mask] = rgb

        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def main():
    parser = argparse.ArgumentParser(
        description="Orca hand ROM point cloud test (MuJoCo)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of random joint configurations to sample (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: orcahand_description/rom_test/output)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Do not write .ply file (only .npy)",
    )
    parser.add_argument(
        "--color-scheme",
        type=str,
        default="blue_pink",
        choices=["blue_pink", "green_yellow"],
        help=(
            "Per-joint color range for .ply: "
            "'blue_pink' (default) or 'green_yellow' (alternative)."
        ),
    )
    parser.add_argument(
        "--print-tip-positions",
        action="store_true",
        help="Run one FK step and print body origin vs tip position for each tracked body (then exit).",
    )
    parser.add_argument(
        "--use-mesh-tips",
        action="store_true",
        help="Compute tip from mesh (furthest point along body +Z) instead of ROM_BODY_TIP_OFFSETS.",
    )
    args = parser.parse_args()

    model_path = get_model_path()
    out_dir = args.output or os.path.join(get_rom_test_dir(), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model:", model_path)
    model = load_model(model_path)
    data = mujoco.MjData(model)

    joint_specs = get_joint_limits_and_qpos_indices(model)
    body_ids, body_names_found = get_body_ids_and_names(model, ROM_BODY_NAMES)
    if not body_ids:
        print("Error: no bodies found for ROM tracking. Check body names.")
        sys.exit(1)
    if not joint_specs:
        print("Error: no limited joints found.")
        sys.exit(1)

    tip_offsets = [np.asarray(ROM_BODY_TIP_OFFSETS.get(n, [0, 0, 0]), dtype=np.float64) for n in body_names_found]

    if args.print_tip_positions:
        mujoco.mj_forward(model, data)
        if args.use_mesh_tips:
            print("Body origin (xpos) vs mesh-derived tip (furthest along body +Z) at default pose:")
        else:
            print("Body origin (xpos) vs tip position (xpos + R @ offset) at default pose:")
        for i, (bid, name) in enumerate(zip(body_ids, body_names_found)):
            origin = data.xpos[bid].copy()
            R = np.array(data.xmat[bid]).reshape(3, 3)
            if args.use_mesh_tips:
                tip = compute_tip_from_mesh(model, data, bid, direction="body_z")
                off_local = R.T @ (tip - origin)
                print(f"  {name}: origin {origin}  tip {tip}  off_local {off_local}")
            else:
                tip = origin + R @ tip_offsets[i]
                print(f"  {name}: origin {origin}  tip {tip}  offset {tip_offsets[i]}")
        return

    print(f"Joints with limits: {len(joint_specs)}")
    print(f"Bodies tracked: {len(body_ids)} ({body_names_found})")
    print(f"Sampling {args.samples} configurations...")

    rng = np.random.default_rng(args.seed)
    points, labels = run_rom_sampling(
        model, data, joint_specs, body_ids, args.samples, rng,
        tip_offsets=tip_offsets, use_mesh_tips=args.use_mesh_tips,
    )

    npy_path = os.path.join(out_dir, "rom_pointcloud.npy")
    save_pointcloud_npy(npy_path, points)
    print(f"Saved point cloud: {npy_path}  shape={points.shape}")

    if not args.no_ply:
        ply_path = os.path.join(out_dir, "rom_pointcloud.ply")
        save_pointcloud_ply(ply_path, points, labels, color_scheme=args.color_scheme)
        print(f"Saved PLY: {ply_path}")

    print(f"Point cloud stats: min={points.min(axis=0)}, max={points.max(axis=0)}")
    print("Done.")


if __name__ == "__main__":
    main()
