"""
Run orcahand_right with a simple policy: move one joint in one direction, then the other.
Uses the fixed-base scene (scene_right.xml).
"""
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PACKAGE_ROOT / "scene_right.xml"

# Policy: which actuator to move and targets (within joint ctrl range)
ACTUATOR_NAME = "right_wrist_actuator"  # wrist pitch
TARGET_LOW = -0.3   # rad
TARGET_HIGH = 0.4   # rad
HALF_PERIOD = 0.50   # seconds per half-cycle (one direction)


def policy_target(time: float) -> float:
    """Target position for the actuated joint: oscillate between TARGET_LOW and TARGET_HIGH."""
    half = HALF_PERIOD
    t = time % (2 * half)  # one full cycle
    if t < half:
        # Linear blend from LOW to HIGH
        return TARGET_LOW + (TARGET_HIGH - TARGET_LOW) * (t / half)
    else:
        # Linear blend from HIGH to LOW
        return TARGET_HIGH + (TARGET_LOW - TARGET_HIGH) * ((t - half) / half)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAME)
    n_act = model.nu
    ctrl0 = np.zeros(n_act, dtype=np.float64)
    ctrl0[act_id] = TARGET_LOW  # start at low so first motion is toward high
    data.ctrl[:] = ctrl0

    print(f"Orca hand right: oscillating '{ACTUATOR_NAME}' (low={TARGET_LOW}, high={TARGET_HIGH}).")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            time = data.time
            data.ctrl[act_id] = policy_target(time)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
