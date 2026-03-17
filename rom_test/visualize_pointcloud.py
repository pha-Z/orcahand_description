"""
Visualize one or more ROM point clouds.

Usage (from repo root, after running run_rom_test.py):
  # Single cloud (default behaviour)
  python orcahand_description/rom_test/visualize_pointcloud.py [path/to/pointcloud.npy or .ply]

  # Overlay two (or more) clouds
  python orcahand_description/rom_test/visualize_pointcloud.py [path/to/pointcloud1.npy or .ply] [path/to/pointcloud2.npy or .ply]

If no path is given, uses:
  orcahand_description/rom_test/output/rom_pointcloud.npy (or .ply if present)
"""

import os
import sys


def load_point_cloud(path):
    """Load a point cloud from .npy or .ply, preserving any stored colors."""
    try:
        import open3d as o3d
        import numpy as np
    except ImportError:  # pragma: no cover - visualization only
        print("Install open3d to visualize: pip install open3d")
        sys.exit(1)

    if path.lower().endswith(".ply"):
        pcd = o3d.io.read_point_cloud(path)
    else:
        points = np.load(path)
        if points.ndim == 2 and points.shape[1] >= 3:
            points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    return pcd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_npy = os.path.join(script_dir, "output", "rom_pointcloud.npy")
    default_ply = os.path.join(script_dir, "output", "rom_pointcloud.ply")

    # Accept multiple paths to overlay several datasets.
    paths = sys.argv[1:]
    if not paths:
        # Fallback to default single cloud.
        if os.path.isfile(default_ply):
            paths = [default_ply]
        else:
            paths = [default_npy]

    for p in paths:
        if not os.path.isfile(p):
            print(f"File not found: {p}")
            print("Run run_rom_test.py first to generate the point cloud(s).")
            sys.exit(1)

    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover - visualization only
        print("Install open3d to visualize: pip install open3d")
        sys.exit(1)

    pcs = []
    for idx, path in enumerate(paths):
        pcd = load_point_cloud(path)
        print(f"Loaded {len(pcd.points)} points from {path}")
        pcs.append(pcd)

    o3d.visualization.draw_geometries(
        pcs,
        window_name="Orca hand ROM point clouds (overlay)",
    )


if __name__ == "__main__":
    main()
