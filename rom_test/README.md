# Orca Hand Range of Motion (ROM) Point Cloud Test

This folder contains a **Range of Motion** test for the Orca hand that builds a **point cloud** of the hand’s reachable workspace using **MuJoCo**. For many random joint configurations within limits, it runs forward kinematics and records the 3D positions of the palm and fingertips. The result is a point cloud you can save (`.npy`, `.ply`) and visualize.

## Requirements

- Python 3.8+
- MuJoCo (pip package)
- Orca hand MJCF and mesh assets (STL files) in `orcahand_description`

## Setup

From the **repository root**:

```bash
pip install -r orcahand_description/rom_test/requirements.txt
```

Ensure the Orca hand MJCF and its mesh paths are correct (see main repo’s orcahand_description). The script loads the **scene** file (which includes the hand and default geom classes):

`orcahand_description/scene_right_extended.xml`

That scene pulls in the right-hand MJCF and the options that define the `bone` / `skin` / `collision` geom classes. Mesh paths in the hand MJCF are relative to the hand file’s directory (e.g. `mjcf/right/visual/...`).

## Run the ROM test

From the **repository root**:

```bash
python orcahand_description/rom_test/run_rom_test.py
```

Options:

- `--samples N` – number of random joint configurations (default: 2000)
- `--output DIR` – output directory (default: `orcahand_description/rom_test/output`)
- `--seed N` – random seed (default: 42)
- `--no-ply` – only save `.npy`, do not save `.ply`

Example:

```bash
python orcahand_description/rom_test/run_rom_test.py --samples 5000 --output orcahand_description/rom_test/output
```

Outputs:

- `rom_pointcloud.npy` – point cloud as `(N, 3)` float32
- `rom_pointcloud.ply` – same point cloud in PLY format (if open3d is installed and `--no-ply` not used)

## View hand with tracked tip markers

To see the hand in the default pose with **red sphere markers** at the tracked tip positions (same points used for the ROM point cloud):

```bash
python orcahand_description/rom_test/view_hand_with_tips.py
```

This opens the MuJoCo viewer; tip positions update if you move joints in the UI. To use **mesh-derived** tips (furthest point on each body’s mesh along its +Z axis) instead of the fixed offsets in `ROM_BODY_TIP_OFFSETS`:

```bash
python orcahand_description/rom_test/view_hand_with_tips.py --use-mesh-tips
```

For the ROM test, use `--use-mesh-tips` to build the point cloud from these mesh-derived tips:

```bash
python orcahand_description/rom_test/run_rom_test.py --use-mesh-tips
```

## Visualize the point cloud

After generating the point cloud:

```bash
python orcahand_description/rom_test/visualize_pointcloud.py
```

Or pass a path explicitly:

```bash
python orcahand_description/rom_test/visualize_pointcloud.py orcahand_description/rom_test/output/rom_pointcloud.ply
```

Requires `open3d` (in `requirements.txt`).

## What the test does

1. Loads the Orca hand MuJoCo model.
2. Collects all joints that have position limits and their qpos indices.
3. For `N` samples (e.g. 2000):
   - Samples random joint positions within limits.
   - Runs `mj_forward` to update kinematics.
   - Records world positions of: `right_palm`, `right_thumb_dp`, `right_index_ip`, `right_middle_ip`, `right_ring_ip`, `right_pinky_ip`.
4. Concatenates all positions into one point cloud and saves it.

The point cloud represents the reachable workspace of those key points over the hand’s range of motion.

## Using the point cloud elsewhere

- **NumPy**: `points = np.load("output/rom_pointcloud.npy")` → shape `(N, 3)`.
- **Open3D**: `pcd = o3d.io.read_point_cloud("output/rom_pointcloud.ply")`.
- You can compare this ROM point cloud to real motion capture or to another simulator (e.g. PyBullet) by running a similar pipeline there and comparing `.npy`/`.ply` files.

## PyBullet alternative

If you prefer PyBullet (e.g. to match another student’s setup), you would:

1. Load the Orca hand URDF in PyBullet.
2. For many random joint configurations within limits, set joint states and compute link world positions (e.g. `getLinkState`).
3. Stack positions into a point cloud and save as `.npy`/`.ply`.

The idea is the same; only the engine (MuJoCo vs PyBullet) and API differ. This project uses MuJoCo so you don’t need to install PyBullet unless you want a second implementation.
