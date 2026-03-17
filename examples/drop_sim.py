"""
Drop the right Orca hand from a random position and orientation.
Reuses orcahand_right.mjcf and adds a free joint at runtime.
"""
import random
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PACKAGE_ROOT / "scene_right.xml"

# MJCF patching (free joint)
FREE_JOINT_NAME = "right_hand_float"
HAND_PATCHED_MJCF = (
    (PACKAGE_ROOT / "models" / "mjcf" / "orcahand_right.mjcf")
    .read_text(encoding="utf-8")
    .replace(
        '<body euler="0.0 0.0 0.0" name="right_tower" pos="0.04 0.0 0.04575">\n      <inertial', 
        f'<body euler="0.0 0.0 0.0" name="right_tower" pos="0.04 0.0 0.04575">\n      <freejoint name="{FREE_JOINT_NAME}"/>\n      <inertial', 
        1
    )
    .encode('utf-8')
)
HAND_VFS_PATH = "models/mjcf/orcahand_right.mjcf"  # Path that scene_right.xml uses in its <include> for the orcahand_right that we want to override

# Drop region (m): axis-aligned box (x, y, z)
SCENE_PARAMS = {
    "drop_lower": (-0.15, -0.15, 1.0),
    "drop_upper": (0.15, 0.15, 2.0),
}


def random_position(
    rng: random.Random,
    lower: tuple[float, float, float] | np.ndarray,
    upper: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Random position (x, y, z) uniformly in the axis-aligned box [lower, upper]."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return np.array(
        [rng.uniform(lo[i], hi[i]) for i in range(3)],
        dtype=np.float64,
    )

def random_unit_quaternion(rng: random.Random) -> np.ndarray:
    """Uniform random unit quaternion (w, x, y, z)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], dtype=np.float64)
    return q / np.linalg.norm(q)


def main() -> None:
    assets = {HAND_VFS_PATH: HAND_PATCHED_MJCF}  # Override orcahand included from scene with patched orcahand
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH), assets=assets)
    data = mujoco.MjData(model)

    # Free joint qpos: 3 position + 4 quat (w,x,y,z) = 7
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT_NAME)
    adr = model.jnt_qposadr[joint_id]
    pos_len, quat_len = 3, 4

    rng = random.Random()
    data.qpos[adr : adr + pos_len] = random_position(rng, SCENE_PARAMS["drop_lower"], SCENE_PARAMS["drop_upper"])
    data.qpos[adr + pos_len : adr + pos_len + quat_len] = random_unit_quaternion(rng)

    mujoco.mj_forward(model, data)

    print("Dropping right hand from random pose.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
