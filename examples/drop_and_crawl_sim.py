"""
Drop the right Orca hand from a random position and orientation, then run a
finger-crawling policy.
Reuses orcahand_right.mjcf and adds a free joint at runtime (like drop_sim.py).
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
        1,
    )
    .encode("utf-8")
)
HAND_VFS_PATH = "models/mjcf/orcahand_right.mjcf"

# Drop: position (m) and orientation. Pitch and yaw fixed at 0; only belly-direction tilt (roll) varies.
DROP_PARAMS = {
    "drop_lower": (-0.15, -0.15, 1.0),
    "drop_upper": (0.15, 0.15, 2.0),
    # Belly-down tilt (rad): roll in [lower, upper]. −π/2 = flat belly down; stay above −π to avoid back.
    "belly_roll_lower": -np.pi / 2.0 - 0.15,
    "belly_roll_upper": -np.pi / 2.0 + 0.25,
}

# Policy
# Finger-crawl: MCP + PIP in sync, wave motion pinky → ring → middle → index
CRAWL_MCP_ACTUATOR_NAMES = [
    "right_pinky_mcp_actuator",
    "right_ring_mcp_actuator",
    "right_middle_mcp_actuator",
    "right_index_mcp_actuator",
]
CRAWL_PIP_ACTUATOR_NAMES = [
    "right_pinky_pip_actuator",
    "right_ring_pip_actuator",
    "right_middle_pip_actuator",
    "right_index_pip_actuator",
]
CRAWL_OPEN = 0.0    # rad (MCP and PIP extended)
CRAWL_CLOSED = 0.9  # rad (MCP and PIP curled)
CRAWL_PERIOD = 1.0  # seconds for one full wave cycle
# Fraction of cycle spent holding at top/bottom (so all fingers finish before reversing)
CRAWL_HOLD_FRAC = 0.25  # 0.25 = hold 25% at bottom, 25% at top; rest is wave down/up

# Wrist: part of crawl, same period; phase shifted 1/4 cycle earlier than fingers
WRIST_ACTUATOR_NAME = "right_wrist_actuator"
WRIST_LOW = -0.6   # rad
WRIST_HIGH = 0.8   # rad
WRIST_PHASE_LEAD = 0.05  # fraction of CRAWL_PERIOD that wrist leads fingers (quarter period earlier)


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


def _quat_mul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Multiply quaternions (w, x, y, z)."""
    wa, xa, ya, za = qa[0], qa[1], qa[2], qa[3]
    wb, xb, yb, zb = qb[0], qb[1], qb[2], qb[3]
    return np.array(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dtype=np.float64,
    )


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles (rad) intrinsic XYZ to unit quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qx = np.array([cr, sr, 0.0, 0.0], dtype=np.float64)
    qy = np.array([cp, 0.0, sp, 0.0], dtype=np.float64)
    qz = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    q = _quat_mul(_quat_mul(qx, qy), qz)
    return q / np.linalg.norm(q)


def random_euler_quaternion(
    rng: random.Random,
    lower: tuple[float, float, float] | np.ndarray,
    upper: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Random unit quaternion (w, x, y, z) from Euler (roll, pitch, yaw) in rad, sampled in [lower, upper]."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    roll = rng.uniform(lo[0], hi[0])
    pitch = rng.uniform(lo[1], hi[1])
    yaw = rng.uniform(lo[2], hi[2])
    return euler_xyz_to_quat(roll, pitch, yaw)


def policy_crawl(time: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Crawl policy: MCP + PIP wave (down, hold, up, hold) and wrist (same period, phase shifted 1/4 earlier)."""
    n = len(CRAWL_MCP_ACTUATOR_NAMES)
    hold = CRAWL_HOLD_FRAC
    wave = (1.0 - 2.0 * hold) / 2.0  # fraction for one wave (down or up)
    if wave <= 0:
        wave = 0.2
        hold = 0.3
    phase = (time / CRAWL_PERIOD) % 1.0
    # Fingers: [0, wave] down, [wave, wave+hold] hold bottom, [wave+hold, wave+hold+wave] up, [wave+hold+wave, 1] hold top
    mcp_targets = np.empty(n, dtype=np.float64)
    pip_targets = np.empty(n, dtype=np.float64)
    for i in range(n):
        if phase < wave:
            s_local = np.clip(phase * n / wave - i, 0.0, 1.0)
            pos = CRAWL_OPEN + (CRAWL_CLOSED - CRAWL_OPEN) * s_local
        elif phase < wave + hold:
            pos = CRAWL_CLOSED
        elif phase < wave + hold + wave:
            p_up = phase - (wave + hold)
            s_local = np.clip(p_up * n / wave - i, 0.0, 1.0)
            pos = CRAWL_CLOSED + (CRAWL_OPEN - CRAWL_CLOSED) * s_local
        else:
            pos = CRAWL_OPEN
        mcp_targets[i] = pos
        pip_targets[i] = pos
    # Wrist: same CRAWL_PERIOD, phase shifted by WRIST_PHASE_LEAD (quarter period earlier)
    wrist_phase = (phase - WRIST_PHASE_LEAD) % 1.0
    half = 0.5
    if wrist_phase < half:
        wrist_target = WRIST_LOW + (WRIST_HIGH - WRIST_LOW) * (wrist_phase / half)
    else:
        wrist_target = WRIST_HIGH + (WRIST_LOW - WRIST_HIGH) * ((wrist_phase - half) / half)
    return mcp_targets, pip_targets, wrist_target


def main() -> None:
    assets = {HAND_VFS_PATH: HAND_PATCHED_MJCF}
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH), assets=assets)
    data = mujoco.MjData(model)

    # Free joint qpos: 3 position + 4 quat (w,x,y,z) = 7
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT_NAME)
    adr = model.jnt_qposadr[joint_id]
    pos_len, quat_len = 3, 4

    rng = random.Random()
    data.qpos[adr : adr + pos_len] = random_position(
        rng, DROP_PARAMS["drop_lower"], DROP_PARAMS["drop_upper"]
    )
    roll = rng.uniform(DROP_PARAMS["belly_roll_lower"], DROP_PARAMS["belly_roll_upper"])
    data.qpos[adr + pos_len : adr + pos_len + quat_len] = euler_xyz_to_quat(roll, 0.0, 0.0)

    crawl_mcp_act_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in CRAWL_MCP_ACTUATOR_NAMES],
        dtype=np.int32,
    )
    crawl_pip_act_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in CRAWL_PIP_ACTUATOR_NAMES],
        dtype=np.int32,
    )
    wrist_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, WRIST_ACTUATOR_NAME)
    data.ctrl[crawl_mcp_act_ids] = CRAWL_OPEN
    data.ctrl[crawl_pip_act_ids] = CRAWL_OPEN
    data.ctrl[wrist_act_id] = WRIST_LOW

    mujoco.mj_forward(model, data)

    print("Drop and crawl: right hand from random pose, wrist + finger-crawling wave.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mcp_t, pip_t, wrist_t = policy_crawl(data.time)
            data.ctrl[crawl_mcp_act_ids] = mcp_t
            data.ctrl[crawl_pip_act_ids] = pip_t
            data.ctrl[wrist_act_id] = wrist_t
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
