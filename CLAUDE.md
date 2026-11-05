# CLAUDE.md

## What this repo is

`hexapod_noetic`: a ROS 1 Noetic catkin package that runs RL policies trained in **hexasim**
(`/home/ubuntu/gitrepos/hexasim`, Isaac Lab + rsl_rl PPO) on the real **Interbotix PhantomX Mark IV**
(`pxmark4`: 18 AX-12A servos, 6 legs × coxa/femur/tibia). This repo only does deployment. Training, reward work and
the observation/action definitions live in hexasim, and **hexasim is the source of truth**. Read
`hexasim/CLAUDE.md` before porting a policy.

State (2026-11): only a skeleton exists. `scripts/policy_runner.py` inits a node, and nothing else is implemented.
`CMakeLists.txt` installs `config/ launch/ models/`, which don't exist yet, so create them before `catkin_make install`.

## Target platform

- Raspberry Pi 4 (aarch64), Ubuntu 20.04, ROS Noetic, **Python 3.8** (no `match`, no `X | Y` types, no
  `list[int]` at runtime). Use NumPy + `onnxruntime` (CPU) and **no torch on the robot**.
- Vendor stack: `interbotix_ros_crawlers` / `interbotix_xshexapod_control`
  (`roslaunch interbotix_xshexapod_control xshexapod_control.launch robot_model:=pxmark4`).
  - Reads `/pxmark4/joint_states` (`sensor_msgs/JointState`). **Its default publish rate is 30 Hz**
    (`config/pxmark4.yaml: joint_state_publisher.update_rate`), which is below the 50 Hz control rate. Raise it, or
    read the servos directly.
  - Writes `/pxmark4/commands/joint_group` (`interbotix_xs_msgs/JointGroupCommand`, `name` = group, `cmd` = rad).
    Groups are per leg (`left_back`, …). Check for or create an `all` group so one message covers all 18 joints.
  - The vendor `joint_order` is **LB, LM, LF, RF, RM, RB** (coxa, femur, tibia each). This is **not** the sim order.
- Bus requirements (hexasim `VP-JEPA_Architecture.md` §6.3): Return Delay Time 0, FTDI latency timer 1 ms, 1 Mbps,
  read Present Position only. 50 Hz is the design point.
- **No IMU on the robot yet** (checked 2026-11-05: no USB IMU, empty I2C bus 1, no IIO devices). Every policy needs
  the 9 IMU dims (accel, gyro, gravity), so the sensor and its frame/convention are still open.
- Pi 4 runs at 1.5 GHz with the ondemand governor. onnxruntime 1.16.3 is installed (`pip3 --user`). Passwordless sudo works.
- Access, network tunnel and the rsync command are in `HARDWARE_ACCESS.md`. That file is gitignored local notes:
  never commit it, and never copy credentials out of it.

## The sim ↔ robot contract (verify against hexasim code for the exact checkpoint you port)

Policies are stage-specific, so check the actor input width before anything else
(`hexasim/source/hexasim/hexasim/rl/load_policy.py: TASK_BY_DIM`, `mdp/layout.py`):
75 Flat (not deployable: uses ground-truth velocity) · 114 S0a · **117 S0b (= π_S0, the first deployable target)** ·
1,125 S1 / 1,269 S2+ (these need the TCN aux learner E_p/V and terrain/vision blocks; the zero-held blocks must be
fed exactly 0).

**Joint order.** Sim order is **grouped by type**: `[coxa×6, femur×6, tibia×6]`, legs in
`LEG_NAMES = (RF, RM, RB, LF, LM, LB)` order (`rl/s1/kinematics.py`, asserted in `hexapod_env.py`). The vendor order
is per leg, LB first. Map to and from `JointState.name` **by name, never by index**.

**Control.** 50 Hz (sim dt 1/200, decimation 4). Target `q_cmd = q_default + 0.5·a`, clipped per joint
(rad): coxa ±0.9599, femur [−1.7104, 1.5708], tibia [−0.7854, 2.1817]. `q_default` from S0a on
(`PXMARK4_AX12A_CFG`) is coxa 0, femur +30°, tibia **+30°**. Phase 1 / Flat used tibia −30°. Deploy deterministic
actions (the Gaussian mean). The servos are position-mode AX-12A (P-only compliance). Do not add a Kd or a profile
the sim did not model: the vendor groups set `profile_velocity 1000 / profile_acceleration 250`, so check what the
AX firmware actually applies.

**o_p (111-D), in this order** (`hexapod_env.py: refresh_proprio`, `mdp/proprio.py`):
1. `q − q_default` (18), with q quantised to the AX-12A tick `300°/1023`
2. `q̇` (18) = `(q_t − q_{t−2}) / (2·0.02)`. Hold the last valid q̇ if either endpoint was a dropped read. No filter.
3. `q_cmd − q` (18), where q_cmd is the **processed** (scaled + offset + clipped) target from the previous step
4. last action (18): the raw policy output `a`, not scaled
5. IMU linear accel, body frame (3): Isaac Lab `ImuCfg` on base_link. **Check whether the installed Isaac Lab 3.0 IMU includes gravity** (the code doesn't say). It must match
   the real IMU's convention
6. IMU gyro, body frame (3)
7. projected gravity, body frame (3): unit vector, ≈ (0, 0, −1) when level
8. clock (12): `(sin φ_i, cos φ_i)` per leg in LEG_NAMES order. Tripod offsets `[0, π, 0, π, 0, π]`,
   `f = clamp(v_cmd_x / 0.20, 0.8, 2.5)` Hz, and **all zeros while standing / zero command**
9. read-validity mask (18): 1 = fresh read, 0 = dropped (with q zero-order-held)

S0b then appends `V_0` (3, the velocity estimate from an online MLP over a 5-frame o_p history. The env owns it and
it is stored in the checkpoint) and `twist_cmd` (3: v_x, v_y, ω_z; trained with v_x ∈ [0.05, 0.20] m/s).
Port `mdp/proprio.py: ProprioReader` 1:1 to NumPy. The spec requires one fill implementation shared by sim and robot.

**Normalisation.** The obs normaliser lives in the checkpoint (`actor_state_dict["obs_normalizer.*"]`).
hexasim `scripts/rsl_rl/play.py` exports `exported/policy.onnx` with the normaliser folded in. Confirm it is folded
in before also normalising on the robot (normalising twice is a silent bug). Extra env-owned nets (V_0, S1 aux) are
**not** in that export, so they need their own export.

**S1+ (blind S1H4) inference.** The ê gate is open when `s1_state.gate_iter ≥ 0`. V_0 (`v0_drop_p` 1) and the clock
(`clock_drop_p` 1) are fully dropped, so feed zeros. ê = `[v_kin + Δv̂, f̂]`, and z_p/ê go into the 1,269-D obs at
[117:181] and [181:196]. Feed every other non-o_p block as exactly 0. The aux net's frame normaliser
(`aux_state_dict.frame_mean/std`) is separate from the actor's.

**Measured Pi 4 latency (2026-11-05, `s1h4x_B0_seed43/model_10998.pt`, compute only, no ROS or bus):**
- Naive path (full W=50 TCN in ONNX + NumPy v_kin + actor): 11.3 ms mean, which does not fit 100 Hz.
- Fused single ONNX graph (TCN evaluated on the last step's receptive cone only, v_kin in-graph with a closed-form
  plane fit, actor with the normaliser folded in): **6.4 ms with 1 thread, 5.5 ms with 2** (p99 < 6.6 ms, max error
  vs the torch modules 3e-6). 4 threads is worse (contention).
- The bus adds about 5 ms per step (18 position reads + SYNC_WRITE, spec §6.3), so a serial 100 Hz loop doesn't fit.
- **A checkpoint trained at 50 Hz must run at 50 Hz.** q̇'s 2Δt, the W=50 window (1 s), the actuator latency and
  per-step action deltas all assume 20 ms. 100 Hz needs retraining in hexasim (decimation 2).

## Porting checklist

1. Pick the checkpoint and stage. Export the ONNX file plus a contract file (joint name order, q_default, clip
   limits, scale, dt, obs layout and widths) into `models/` / `config/`. Generate the contract from hexasim code,
   don't hand-type it.
2. Parity test offline before the robot: record sim obs → action pairs (hexasim, deterministic) and check that the
   NumPy obs builder + onnxruntime reproduce them (< 1e-5).
3. On hardware: start suspended or lying down, ramp from `sleep_positions` to `q_default`, then enable the policy with
   zero command. Have an e-stop/torque-off path and clamp every command to the clip limits.
4. Log raw obs, actions, loop timing and dropped reads on one monotonic clock. Report loop p99 latency.

Unresolved sim-to-real items (hexasim marks these "provisional / to be measured"): the AX-12A model constants,
the drop-rate vs load model, IMU noise/bias, and joint sign conventions vs the vendor driver (same URDF source, but
verify each joint's sign by name on the stand).

## Build

```bash
cd ~/interbotix_ws && catkin_make --pkg hexapod_noetic && source devel/setup.bash
```
