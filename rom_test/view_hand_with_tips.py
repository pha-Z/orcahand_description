"""
Show the Orca hand in default pose with sphere markers at the tracked tip positions.

Usage (from repo root):
  python orcahand_description/rom_test/view_hand_with_tips.py
  python orcahand_description/rom_test/view_hand_with_tips.py --use-mesh-tips

Uses the same ROM_BODY_NAMES and ROM_BODY_TIP_OFFSETS as run_rom_test.py.
Tip positions update if you change joint angles in the viewer (e.g. via the joint slider UI).
--use-mesh-tips: show mesh-derived tip (furthest point along body +Z) instead of offset-based.
"""

import argparse
import os
import sys

import numpy as np
import mujoco
import mujoco.viewer

# Reuse config from run_rom_test so tips match what we track in the ROM
from run_rom_test import (
    get_model_path,
    load_model,
    get_body_ids_and_names,
    ROM_BODY_NAMES,
    ROM_BODY_TIP_OFFSETS,
    collect_body_positions,
)


def main():
    parser = argparse.ArgumentParser(description="View Orca hand with tip markers")
    parser.add_argument(
        "--use-mesh-tips",
        action="store_true",
        help="Compute tip from mesh (furthest along body +Z) instead of offset.",
    )
    args = parser.parse_args()

    model_path = get_model_path()
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model = load_model(model_path)
    data = mujoco.MjData(model)
    body_ids, body_names_found = get_body_ids_and_names(model, ROM_BODY_NAMES)
    if not body_ids:
        print("No bodies found for ROM_BODY_NAMES.")
        sys.exit(1)

    tip_offsets = [
        np.asarray(ROM_BODY_TIP_OFFSETS.get(n, [0, 0, 0]), dtype=np.float64)
        for n in body_names_found
    ]
    mujoco.mj_forward(model, data)

    # Sphere radius for tip markers (meters)
    tip_sphere_radius = 0.005
    # Bright red so they stand out
    tip_rgba = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)
    n_tips = len(body_ids)
    id_eye = np.eye(3).astype(np.float64)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Recompute tip positions from current pose (so sliders update tips)
            tips = collect_body_positions(
                data, body_ids, tip_offsets, model=model, use_mesh_tips=args.use_mesh_tips
            )

            viewer.user_scn.ngeom = n_tips
            for i in range(n_tips):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([tip_sphere_radius, 0, 0], dtype=np.float64),
                    pos=tips[i].astype(np.float64),
                    mat=id_eye.flatten(),
                    rgba=tip_rgba,
                )
            viewer.sync()


if __name__ == "__main__":
    main()
