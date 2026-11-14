# Hardware progress

Chronological log of porting hexasim policies to the real PhantomX Mark IV (`pxmark4`). The newest entry goes at the
bottom. Each entry gives the command or checkpoint it used, so its numbers can be reproduced.

## 2026-11-05: Deployment context, IMU survey, Pi 4 latency of the blind S1H4 policy

### Context established
- This repo (`hexapod_noetic`) is a ROS Noetic skeleton: `scripts/policy_runner.py` only inits a node. `CLAUDE.md`
  holds the sim↔robot contract taken from hexasim. That covers the joint order, the action mapping and clip limits,
  the 111-D `o_p` frame, and the S1+ latent slots.
- **Joint order:** sim = `[coxa×6, femur×6, tibia×6]`, legs RF RM RB LF LM LB (`hexasim rl/s1/kinematics.py`). The
  vendor `pxmark4.yaml` order is per leg: LB LM LF RF RM RB. Map by name.
- The vendor `joint_state_publisher.update_rate` is 30 Hz, below the 50 Hz control rate, so it must be raised or the
  servos read directly.
- **Control rate: 50 Hz** (user confirmed 2026-11-05; the earlier "100 Hz" was a mix-up). This matches training
  (sim dt 1/200, decimation 4).

### IMU survey on the Pi (result: none present)
Commands ran on the Pi over the tunnel (`ssh -p 2222 pibot@127.0.0.1`):
- `lsusb`: FT232H (servo bus), USB hubs, Dell keyboard/mouse. No USB IMU.
- I2C bus 1, scanned as root with read-only probes over 0x03–0x77 (Python `fcntl` `I2C_SLAVE` + 1-byte read,
  because `i2c-tools` isn't installed): **no devices**.
- `/sys/bus/iio/devices` is absent, there are no IMU ROS packages, and `/dev/spidev0.{0,1}` exist (SPI is enabled,
  nothing identified on it).

**Decision (user): add an MPU6050.** It gives accelerometer + gyro (6-axis, no magnetometer). The policy's 9 IMU dims
are accel (3), gyro (3) and projected gravity (3). Open items for the MPU6050 integration:
- Projected gravity is not measured directly. It needs an attitude estimate (for example a complementary or Madgwick
  filter on accel + gyro) expressed as a unit vector in the body frame, ≈ (0, 0, −1) when level.
- Mounting frame → base_link frame rotation (axis signs) must match the sim body frame (x forward, y left, z up).
- Accel convention: check whether hexasim's Isaac Lab `ImuCfg` `lin_acc_b` includes gravity (not confirmed from the
  installed code), and match the MPU6050 output to it.
- Characterise noise and bias against the sim's `imu_acc_std` / `imu_gyro_std` and bias DR in hexasim `hexapod_env.py`.
- Sample rate ≥ 50 Hz, synchronised with the joint reads on one monotonic clock.

### Pi 4 latency test: blind S1H4 control path
- **Checkpoint:** `hexasim/logs/rsl_rl/hexapod_s1h4/2026-10-16_21-12-36_s1h4x_B0_seed43/model_10998.pt`
  (task `Hexapod-S1H4-v0`, arm B0, W = 50, actor input 1,269-D, TCN dilations {1, 2, 4, 8}, RF 61, ê gate open
  (`gate_iter` 400), `v0_drop_p` = `clock_drop_p` = 1.0, so V_0 and the clock are fed as zeros).
- **Platform:** Raspberry Pi 4 Model B Rev 1.4, 4 cores, 1.5 GHz, Ubuntu 20.04, Python 3.8.10, numpy 1.22.0,
  onnxruntime 1.16.3 (CPU EP, installed with `pip3 install --user onnxruntime`). Governor: ondemand. The ROS stack
  was not running.
- Scripts are in `tools/s1_export/`. The export scripts run on the workstation in `env_isaaclab` and the bench
  scripts on the Pi. The bundle that was benchmarked is on the Pi at `~/s1h_bench/`.
  ```bash
  # workstation: <out> gets aux.onnx, actor.onnx, control.onnx and the sample-I/O npz files
  python tools/s1_export/export_s1h.py <model.pt> <out>
  python tools/s1_export/export_fused.py <model.pt> <out>
  # bench_pi.py also needs <out>/kin_consts.npz and vkin_ref.npz (written by a one-off snippet in this session,
  # not in a script yet); bench_fused.py needs only control.onnx + fused_io.npz
  cp tools/s1_export/bench_*.py <out>/ && rsync -az -e 'ssh -p 2222' <out>/ pibot@127.0.0.1:~/s1h_bench/
  # Pi
  python3 ~/s1h_bench/bench_fused.py 3000
  ```
  - `export_s1h.py` exports `aux.onnx` (TCN + Δv̂ + f̂) and `actor.onnx` (rsl_rl `as_onnx`, normaliser folded in).
  - `export_fused.py` exports `control.onnx`: inputs `hist (1,50,111)` raw frames and `obs_base (1,1269)`, output
    `action (18)`.
    - The TCN is evaluated only on the receptive cone of the last step: 133 vs 400 conv position-evaluations, with
      the same zero padding, so the output is identical.
    - v_kin runs in-graph, with `lstsq` replaced by the closed-form 3×3 normal equations.
    - z_p goes to obs [117:181] and ê = [v_kin + Δv̂, f̂] to [181:196].
  - `bench_pi.py` / `bench_fused.py` measure parity, per-step latency, and timed 100 Hz / 50 Hz loops (3,000 steps).

**Parity**
- Fused graph vs the original hexasim torch modules (64 random inputs, workstation): max |Δ| z_p 2.0e-6,
  v_kin 4.5e-7, action 3.0e-6.
- On the Pi vs saved references: z 1.1e-6, Δv̂ 5.8e-8, f̂ 1.3e-5, action 3.9e-6 (split path), 6.7e-6 (fused);
  NumPy v_kin 5.5e-6.

**Latency (compute only: no ROS, no servo bus)**

| Path | Threads | Mean | p99 | Max |
|---|---|---|---|---|
| Split: full TCN + NumPy v_kin + actor | 1 | 11.32 ms | 12.21 ms | 13.10 ms |
| Split | 2 | 10.19 ms | 21.24 ms | 24.97 ms |
| Split | 4 | 19.21 ms | 61.77 ms | 90.17 ms |
| **Fused `control.onnx`** | 1 | **6.42 ms** | 6.54 ms | 6.77 ms |
| **Fused `control.onnx`** | 2 | **5.51 ms** | 5.61 ms | 5.72 ms |

- Split-path components at 1 thread: TCN + V 8.18 ms, NumPy v_kin 2.05 ms, actor 0.89 ms.
- Fused, timed loops: 0 / 3,000 deadline misses at both 100 Hz and 50 Hz, with 1 and 2 threads. The performance
  governor made no meaningful difference (it was restored to ondemand afterwards).
- 4 threads causes contention, so use 1–2 threads for inference.
- **Conclusion: 50 Hz is comfortable.** The fused graph takes about 6 ms, plus about 5 ms of bus time (18 position
  reads + SYNC_WRITE, an estimate from hexasim `VP-JEPA_Architecture.md` §6.3, not yet measured), against a 20 ms
  budget.

### Remaining before a first hardware run
- MPU6050 frame alignment (match sim base_link: +x forward, +y left, +z up), static bias calibration, attitude filter for projected gravity.
- Raise the joint-state rate to ≥ 50 Hz (or read the bus directly) and measure real bus time per step.
- A NumPy port of hexasim `mdp/proprio.py` (`ProprioReader`): quantisation, ZOH on dropped reads, two-sample q̇,
  validity mask.
- The ROS runner itself (obs assembly, `control.onnx`, name-mapped `JointGroupCommand`, clip, start-up ramp from
  sleep pose to the rest pose, torque-off path, logging).
- Verify each joint's sign convention by name on the stand.

## 2026-11-05: MPU-6050 hardware attach and live verification

### Wiring and Electrical Safety
- **Device:** MPU-6050 (GY-521 breakout board).
- **Voltage Safety:** BCM2711 GPIOs on Pi 4 are strictly 3.3V only (not 5V tolerant). Breakout VCC was connected to Pi **3.3V (Pin 1)**, not 5V. At ~3.8 mA current draw, LDO dropout is < 40 mV (~3.26V rail), keeping SDA, SCL, and INT strictly within 3.3V logic limits without risk of 5V pull-up leakage.
- **Pinout configured:**
  - VCC -> RPi Pin 1 (3.3V)
  - GND -> RPi Pin 9 (GND)
  - SDA -> RPi Pin 3 (GPIO 2 / SDA1)
  - SCL -> RPi Pin 5 (GPIO 3 / SCL1)
  - AD0 -> RPi Pin 9 / GND (sets I2C address to `0x68`)
  - XDA, XCL -> Floating (auxiliary master bus, unused)

### System Configuration
- Configured persistent udev rule on Pi `/etc/udev/rules.d/99-i2c.rules`:
  ```udev
  KERNEL=="i2c-[0-9]*", GROUP="dialout", MODE="0666"
  ```
  This grants user `pibot` direct access to `/dev/i2c-1` without `sudo`.

### Bus Verification & Live Motion Test
- Bus probe on `/dev/i2c-1` identified device at address `0x68`.
- `WHO_AM_I` register (`0x75`) read confirmed `0x68`.
- Initialized power management (`PWR_MGMT_1 = 0x01` PLL with X-gyro clock) and configured scales: $\pm 4\text{ g}$, $\pm 500^\circ/\text{s}$.
- Burst-read 14 bytes (`0x3B`–`0x48`) in a single I2C transaction over 40 steps (~4 seconds) while moving the sensor:
  - Dynamic acceleration tracked between $0.77\text{ g}$ and $1.35\text{ g}$ (nominal baseline $\|a\| \approx 1.0\text{ g}$).
  - Gyroscope responded dynamically across all 3 axes (rates reaching $-68^\circ/\text{s}$ to $+74^\circ/\text{s}$).
  - Sensor is fully operational and responsive over I2C.

## 2026-11-06: IMU frame, mount and calibration

### What the sim IMU is (hexasim)
- `S0aSceneCfg` has `ImuCfg(prim_path=".../Robot/Geometry/tn__pxmark4base_link_VP")` with no offset, so the IMU frame
  is the `base_link` frame. In `pxmark4.urdf.xacro` that frame has its origin at the centre of the leg footprint (the
  coxa plane), **+x forward** (front legs at x = +0.125), **+y left**, **+z up**. COM = (1.5, 0.3, 15.3) mm. The base mesh
  spans x ±0.142, y ±0.111, z −0.033…+0.080 m. Render: `python3 tools/imu/render_frame.py tools/imu/base_link_frame.png`.
  - Not checked: whether the USD import rotated the `..._VP` mesh prim relative to the URDF frame (needs a sim run that
    reads `lin_acc_b` at rest).
- **The sim accelerometer includes gravity.** Isaac Lab `isaaclab_physx/sensors/imu/imu.py` adds a gravity bias of −g
  ("real IMUs always measure gravity"), so `lin_acc_b` ≈ (0, 0, **+9.81**) when level. This matches the MPU-6050
  convention and closes the open item from 2026-11-05. `projected_gravity_b` ≈ (0, 0, −1) when level.
- Obs construction is `hexapod_env.py:184-186`: `lin_acc_b + bias + N(0, 0.05)`, `ang_vel_b + bias + N(0, 0.02)`,
  `projected_gravity_b + N(0, 0.02)`. Bias DR std: 0.05 m/s² (accel), 0.01 rad/s (gyro).

### Mount
- GY-521 fixed to the rigid frame at **r = (+0.019, 0, +0.044) m** from the `base_link` origin (user measured). The
  board's axes are rotated about 90° about z relative to the body; the calibration solves that, so no alignment by eye.
  Centre of the base was rejected (12 V battery lines nearby, I2C pickup risk); above the Pi was rejected (large z
  offset and heat, which drifts gyro bias).
- Registers (written by `imu_calib.py`): DLPF ~44 Hz, ±8 g (4096 LSB/g), ±500 °/s (65.5 LSB/°/s). The 2026-11-05 test used
  ±4 g. A Python loop with `sleep(0.01)` reads at about 84 Hz.

### Calibration (`tools/imu/imu_calib.py`, copied to `~/imu_calib.py` on the Pi; result in `config/imu_calib.json`)
```bash
python3 ~/imu_calib.py six                     # bare board on each of its 6 faces, auto-detects the face
python3 ~/imu_calib.py capture level 6         # mounted, robot level and still
python3 ~/imu_calib.py capture nose_up 6       # front raised ~30°
python3 ~/imu_calib.py capture left_up 6       # left side raised (yaw check)
python3 ~/imu_calib.py solve 0.019 0 0.044     # writes ~/imu_calib/calib.json; args = lever arm r in metres
python3 ~/imu_calib.py live 10                 # calibrated body-frame accel, accel@origin, gyro, gravity
```
- **Accel offset / scale (six-face):** offset (+0.258, −0.612, −0.705) m/s², scale (1.0051, 1.0078, 0.9895). The offsets
  are about 10× the sim's accel bias DR, so they have to be subtracted. Held-out check: a pose that wasn't used in the fit
  had |a| 9.057 raw and 9.830 corrected (g = 9.807).
- **Rotation `R_body_from_sensor`** (v_body = R · v_sensor): body +x = sensor −y, body +y = sensor +x, body +z = sensor z,
  with about 3° of mount tilt absorbed. Level pose → accel (0, 0, 9.84), gravity (0, 0, −1.004), so projected gravity must be
  normalised to unit length in the runner.
- **Gyro bias (sensor frame):** (0.260, 0.051, 0.009) rad/s. Stable across all captures (scatter 0.0007 rad/s). The
  sim's bias DR is only ±0.01 rad/s, so subtract it, and re-measure for about 2 s at start-up (it drifts with temperature).
- **Checks:** nose-up pose = 31.7° (not yet compared to a measured block height). Left-up pose, body-frame up =
  (−0.009, 0.315, 0.949): y > 0 and x ≈ 0, so the residual yaw error of the forward axis is about 1.6°. After settling
  the robot level again, `live` showed gravity (0.01, 0.02, −1.00), roughly 1° level repeatability on the table.
- **Lever arm:** the accelerometer measures `a_origin + α×r + ω×(ω×r)`, so the runner subtracts the last two terms to refer
  it to the origin as the sim does. `live` implements this with α from a smoothed backward difference of the gyro.

### Sim frame check (USD) and the runner's IMU block
- **USD check:** `hexapod.usda` (the asset hexasim loads) was opened with `pxr`. The IMU prim
  `/pxmark4/Geometry/tn__pxmark4base_link_VP` *is* the base_link rigid body, with an identity transform on it and all its
  ancestors, so the sim IMU frame = the base_link frame with no extra rotation or offset. The USD is Z-up, metres, and
  the leg mounts match the URDF (right_front coxa at (+0.125, −0.065, 0), so front = +x, right = −y). This is a static read
  of the file, not a runtime sim read of `lin_acc_b`.
- **`scripts/imu_block.py`** (NumPy, Python 3.8, no ROS): `MPU6050` driver + `ImuBlock`. `start()` re-measures the gyro bias
  from ~2 s of still samples (rejects the capture if the robot moved or the bias is far from calibration);
  `update(a_raw, w_raw, t)` returns the 9 IMU dims `[accel@origin, gyro, projected gravity]` in the sim's convention:
  calibrated accel/gyro rotated to the body frame, accel referred to the base_link origin (lever arm), and projected gravity
  from a gravity-vector complementary filter (gyro propagation `du/dt = −ω×u`, accel correction gated by `| |a| − g |` so
  it fades while the body accelerates, ~1 s time constant at rest), output normalised to unit length.
- **Offline tests** (`python3 tools/imu/test_imu_block.py`, synthetic motion through the real calibration file, no robot):
  gravity error 0.01° mean / 0.17° max; gyro in body frame exact; accel@origin RMS error 0.022 m/s² with the lever-arm
  correction vs 0.030 without (the synthetic motion is slow, so this improves it only slightly); `start()` recovers a
  drifted gyro bias; the display attitude matches the true rotation to 0.17° over tilts and yaw spins.
- **Motion check** (`tools/imu/imu_check.py`): rotates the accelerometer's "up" vector forward with the gyro over 0.5 s and
  compares it with the accelerometer's next "up". Correct data gives median 0.2° / p95 0.7°; a flipped gyro y axis gives p95
  33.8° and swapped x/y gives p95 23.7°, so it catches a wrong axis, sign or handedness. It also reports whether the robot
  was rotated about all three body axes and tilted nose up/down, left/right, and yawed both ways.
- **Live viewer** (`tools/imu/imu_viz.py`): the URDF body (bounding box from the base mesh, front face red, left face green,
  footprint hexagon through the coxa mounts, leg stubs, body axes, IMU position) is driven by the real attitude. Heading is
  integrated from the gyro (no compass, slow drift). Sources: `--live` (streams from the Pi over the tunnel),
  `--replay FILE`, `--demo`; `--save-png OUT` runs headless. Sessions are saved to `tools/imu/logs/` (gitignored).
  ```bash
  rsync -azR -e 'ssh -p 2222' scripts/imu_block.py tools/imu/imu_stream.py config/imu_calib.json pibot@127.0.0.1:~/imu_tools/
  python3 tools/imu/imu_viz.py --live          # hold the robot still for 2 s, then tilt and spin it; q to finish
  ```
  `imu_stream.py` was run on the Pi under Python 3.8 at 50 Hz with the robot still: start-up bias (0.2596, 0.0509, 0.0082)
  rad/s (matches the calibration), level gravity (0.011, 0.026, −1.000), accel@origin (−0.09, −0.26, 9.85) m/s². The window
  opens on this workstation's display (`DISPLAY=:0`, TkAgg); a demo run of it was smoke-tested.
- **Not done yet:** the live motion check with the robot in hand (the hands-on tilt/spin session on the real sensor).

### Still open for the IMU
- Wire `ImuBlock` into the ROS runner (`scripts/policy_runner.py` is still a skeleton) and run the hands-on motion check.
- Level calibration is on a table; check that projected gravity reads about (0, 0, −1) with the robot standing in the rest
  pose on a flat floor, and fold in any residual tilt.
- Measure accel/gyro noise with the servos powered and the robot walking, against the sim's 0.05 m/s² / 0.02 rad/s.
- Raw capture files (`~/imu_calib/*.npz`) are on the Pi only. `config/imu_calib.json` is the result.

## 2026-11-06: Keyboard teleop + safe policy runner (S1H4 B0)

**Offline work done.** The hardware steps are not started yet.

- **Contract / models** (`tools/policy/export_contract.py`, workstation env_isaaclab). These are generated from
  hexasim code, the run's `params/env.yaml` and the checkpoint `s1h4x_B0_seed43/model_10998.pt` (`s1_state`: gate open,
  `v0_drop_p` 1, `clock_drop_p` 1, W 50). Outputs: `config/contract_s1h4.json` (joint order, q_default, clip, scale 0.5,
  dt 0.02, 1,269-D layout, command ranges, FK constants, servo IDs and sleep pose by name from the vendor
  `pxmark4.yaml`, compliance 1/32, punch 32) and `models/control.onnx` (fused, max error vs the torch modules 3e-6).
  `models/parity_ref.npz` is 120 steps (7.3 % dropped reads) through hexasim's own `ProprioReader`, `S1Aux` and actor,
  with `HexapodS1Env`'s history fill (first frame repeated) and obs assembly.
- **Parity:** the NumPy `ProprioReader` port matches to 1e-6 (q, mask) / 1e-5 (q̇, torch float32). The NumPy
  o_p + history + ONNX path reproduces the reference actions to **1.2e-6**, on the workstation and on the Pi
  (Python 3.8.10, ORT 1.16.3).
- **Pose geometry differs from the plan.** PLANT at r = 12 cm is unreachable: at toe z −3.3 cm the femur clip (−98°)
  and tibia clip (125°) need r ≥ ~15.7 cm, while toe z −17 cm allows r ≤ ~14.3 cm. Stand-up (17.4 s after a 2 s ramp
  to SLEEP, all joints ≤ 1.2 rad/s):
  SLEEP → PLANT (r 16.5, z −3.3) → push-up to toe z −12 cm (WIDE_STAND, belly 8.7 cm) → re-step each leg in to
  r 6.7 cm, **one leg at a time** in tripod order RF, LM, RB, LF, RM, LB → rise to q_default (belly 13.7 cm).
  One leg at a time instead of a tripod: a 3-leg stance at r 16.5 cm puts ~1.1 N·m on each femur (~75 % of stall,
  above the 70 % torque limit). Sit-down = the same path backwards. A fault mid-stand-up reverses from where the robot is.
  Renders: `tools/poses/{sleep,plant,wide_stand,stand,standup_sequence}.png`.
- **Sign check `c`** pulses femur and tibia by **−0.15** rad (foot lifts) instead of +0.15 (which would push the foot
  into the ground while lying down). Coxa uses +0.15.
- **Extra safeguards beyond the plan:** tilt > 60° (fallen) or a dead bus → torque off at once instead of sitting;
  servo status alarm bits (overload/overheat/voltage) → FAULT; battery is the median of the last 18 health reads
  (~0.36 s debounce); third Ctrl-C → immediate exit (torque off in `finally`); the runner refuses to open the bus
  while a vendor `xs_sdk` process runs.
- **Offline tests** `tools/test_teleop_offline.py`: 26/26 pass on the workstation and the Pi (FK/IK + hexasim FK
  match, clip/rate/feet-fixed checks, parity, fake-bus FSM: drop burst, overrun, tilt, NaN, SIGHUP, low V, IMU stale,
  bus dead, fault mid-stand-up, x/freeze, far-pose confirm, sign check, command snapping). `--fake` dry run: period
  p99 20.0 ms, SIGTERM → clean exit.
- **Bus bench attempt:** the servos were unpowered (battery not connected). Every read timed out at 3 ms, so there
  are **no timing numbers yet**. The vendor autostart had launched `xs_sdk`; it exited at boot (no servos), but with
  servo power on at boot it would hold the bus. Stop or disable it before running anything here.

### Next on hardware
1. Battery on, vendor stack stopped → `python3 tools/bus/bus_bench.py`. Decision point: bus+IMU p99 + 6.4 ms
   inference must fit 20 ms.
2. `python3 scripts/hexapod_teleop.py`, lying down: `c` sign check on all 18 → `u`/`j` ×3 (no policy) → `p` at zero
   command for 30 s → v_x 0.05.

## 2026-11-06 (later): The servos are XL430-W250. Driver ported, bus benched

- Both bus-bench runs got no answer. That was **not** the battery: a ping scan found all 18 servos answering only
  **Protocol 2.0** at 1 Mbps, model **1060 = XL430-W250**, firmware 45. Return Delay 0, drive mode 4, operating mode 3
  (position), limits 0..4095, PWM limit 885, position gains P 640 / I 0 / D 3600 (factory), no profile, 11.4–11.6 V.
- Vendor autostart `xshexapod_rpi4_boot.service` is **stopped and disabled** (it ran the joystick stack + xs_sdk at
  boot). Restore with `sudo systemctl enable --now xshexapod_rpi4_boot.service`.
- Driver `scripts/dxl_bus.py` (Protocol 2.0, SYNC_READ / SYNC_WRITE, CRC-16, byte stuffing) replaces the Protocol 1.0
  `ax12_bus.py`. Servo setup (RAM only): P 1061 (AX-12A compliance match), I/D/FF 0, profile 0, Goal PWM 619 (70 %)
  for scripted moves, 885 for the policy. Observations stay on the AX tick grid (proprio.py).
- **Bus bench** (`tools/bus/bus_bench.py --with-policy`, torque off, 1,000 steps, 0 / 18,000 dropped reads):

  | config | inference p99 | step p50 / p99 / max |
  |---|---|---|
  | ondemand, 1 thread | 9.8 ms | 17.6 / 20.9 / 26.9 ms (overruns) |
  | ondemand, 2 threads | 5.6 ms | 14.5 / 16.6 / 17.4 ms |
  | performance, 1 thread | 6.5 ms | 15.5 / 16.8 / 17.7 ms |
  | **performance, 2 threads** (runner default) | 5.5 ms | 14.6 / 15.7 / 16.8 ms |

  Parts at performance/2 threads: SYNC_READ 18 positions 5.3 ms, IMU 1.7 ms, write 1.1 ms, health 0.8 ms (every 2nd
  step, after the write). The runner switches the governor to performance for its session and restores it on exit.
- **Runner start-up on the robot** (torque off, no keyboard, SIGTERM after 12 s): 18 XL430 checks pass, gains applied,
  IMU bias captured, policy loaded, 356 steps at period p99 20.0 ms, 0 dropped reads, clean exit, torque off on all 18,
  governor restored. Offline tests: 28/28 on the workstation and the Pi.

### Next on hardware
1. Robot on its belly: `python3 scripts/hexapod_teleop.py` → `c` sign check (all 18 joints, by name).
2. `u` / `j` three times without the policy. Watch the XL430 tracking with P 1061 / D 0; if it oscillates, try
   `--kp 800`.
3. `p` at zero command for 30 s (loop p99 in the status line) → v_x 0.05.

### SLEEP pose = the robot's relaxed belly-down pose (2026-11-06)
- `u` / `c` were refused: with torque off on the belly the legs relax away from the vendor sleep angles (tibias
  ~1.17 rad instead of 1.57, LF tibia 0.72), beyond the old 0.3 rad check. The check is now per joint type
  (coxa 0.3, femur 0.6, tibia 1.0 rad).
- At the user's request, SLEEP is now the captured pose: `tools/poses/capture_sleep.py` (torque off, median of 25
  reads) → `config/sleep_pose.json`, used by `joint_map.Contract` instead of the vendor `sleep_positions`.
  Delete the file to go back. Tests 29/29.
- **To check:** by URDF FK, this pose puts the toes 0–1.6 cm above the coxa plane, i.e. 3–5 cm above the floor
  (render `tools/poses/sleep.png`). If the real feet rest on the floor instead, there is a ~3 cm joint-zero /
  geometry offset to resolve before standing.

### First policy run (2026-11-06, log `~/hexapod_logs/2026-11-06_09-55-11`)
- Stand-up OK. STAND hold rock-steady (joint std ≤ 0.001 rad, gyro std 0.002 rad/s): the XL430 gains are not the jitter.
- Policy at zero command: within ~2 s the middle coxas went to +0.58 / −0.58 rad and the femurs +0.37 rad, then held
  with vibration (joint velocity sign flips 5–10/s, accel std 0.4–0.8 m/s² vs 0.03 at STAND).
  - **That posture is the policy's own standing pose.** Offline (ideal standing obs) it does the same. In Isaac Sim
    (`record_gait.py --cmd 0.0`, same checkpoint, flat) it stands with RM coxa +32°, LM −28°, femurs 42–54°,
    tibias 26–34° (robot: +33° / −32°, ≈51°, ≈25°). Replaying the logged o_p reproduces the logged actions to 7.7e-7,
    so the observation pipeline is not the cause.
  - The vibration around that pose is still open. Candidates: XL430 P-only (D 0) at 100 % PWM vs the sim's AX model;
    stick-slip of rubber toes on tile while the middle feet re-pose (the sim randomises friction and allows a settle
    step); or the policy dithering in sim too (check `hexasim/logs/videos/gait/deploy_stand_check-step-0.mp4`).
- **Fault:** after `w` (v_x 0.05), sim idx 16/17 (ids 6, 3) dropped > 5 reads in a row → sit-down (worked). Drops were
  always the tail of the SYNC_READ chain: each XL430 waits for the previous reply, so one garbled packet under load
  silences the rest. Fixes: (1) the missing ids are re-read in a second short SYNC_READ; (2) our own select loop
  ends a reply stream after 2 ms of silence. A broken chain now costs 6.0 ms + 3.0 ms retry instead of a 10 ms
  timeout (pyserial's `inter_byte_timeout` has 0.1 s granularity on Linux). The IMU filter's `np.cross` was replaced
  (−0.8 ms per step). Bench after the changes: step p99 15.9 ms, 0 / 18,000 drops (torque off).

## 2026-11-11: Task added — zero-shot test of the sim checkpoints on the real robot (not run yet)

Requested by the user from the hexasim side (hexasim `research/phase2-progress.md` §2.3.34, decisions of 2026-11-11). Every policy so
far was trained on the AX-12A actuator model; the robot has XL430-W250 servos. This test says how far the existing checkpoints
transfer as they are, and gives the bench numbers the XL430 sim model is built from. The sim trainings that are re-run afterwards
depend on its result.

### Tasks
1. **Zero-shot walk, blind policy (S1H4 B0, the checkpoint the runner already loads), flat floor.** Zero command 30 s, then v_x 0.05,
   0.10, 0.15 m/s, ≥ 3 runs of ≥ 20 s each. Record per run: measured speed (tape + stopwatch, or video), falls / sit-down faults,
   dropped reads, loop p99, and the runner log directory. The standing vibration from the 2026-11-06 first run is still open — note
   whether it is present at each command.
2. **Joint tracking from the same logs:** per joint type, `q_cmd − q_meas` RMS and peak in stance and swing, for comparison with the
   sim's AX-12A tracking error on the same checkpoint (hexasim `record_gait.py`).
3. **Servo bench for the motor model (one XL430, unloaded and under a known load):** position step response at the gains the runner
   sets (P 1061, D 0, PWM 885) — rise time, overshoot, steady-state error; no-load speed at the bus voltage; holding torque vs
   tracking error (lever + weight); dropped-read rate vs load. These replace the datasheet numbers that hexasim marks "verify".
4. **Robot numbers the sim needs with the servo swap:** total mass, per-leg mass if a leg can be weighed, bus voltage under load.
5. Optional, only if (1) walks: one low obstacle (≈ 3 cm board) at 0.10 m/s, as a first look at terrain transfer.

Results go into this file as a dated entry (commands, log directories, numbers), then into hexasim's XL430 migration notes.

## 2026-11-11: Policy jitter diagnosed (hung tests + sim comparison) and deploy-side command shaping

Pi now at 192.168.0.106 (tunnel unchanged). Checkpoint: S1H4 B0 `s1h4x_B0_seed43/model_10998.pt`. All tests hung in a harness unless
noted. Logs in `~/hexapod_logs/`: `2026-11-11_09-43-52` (hung, no shaping), `2026-11-11_10-00-32` (hung, 2 onsets, voltage + read
diagnostics), `2026-11-11_11-46-14` (run A), `2026-11-11_11-58-00` (run B). Sim runs: `tools/sim/sim_onset.py` (hexasim env, 64 envs, flat).

### What the jitter is
- **It is the policy's own action chatter, and in sim it is worse.** Standing at cmd 0 (t = 2–5 s), band 6–25 Hz:

  | | joint rms | action rms | accel std | gyro std |
  |---|---|---|---|---|
  | robot, floor 11-06 (after onset) | 0.0011 rad | 0.0026 | 0.10 m/s² | 0.014 rad/s |
  | sim, same checkpoint | 0.0135 rad | 0.063 | 4.8 m/s² | 0.080 rad/s |

  At policy onset (first 25 steps), max |Δa| per step: robot 0.18 (floor) / 0.12 (hung), sim 0.49 for both the training reset (±0.25 rad)
  and a zero-offset reset. The sim looks smooth because the AX-12A model (dead zone, compliance slope, speed cap ≈ 0.07 rad/step)
  and PhysX absorb it. In walking at 0.05 m/s the sim policy commands jumps of 0.24 rad/step at p99, while its joints move ≤ 0.068.
  The only smoothness term in training is `action_rate_l2` at weight −0.01 (no stage overrides it).
- **Hung and floor onsets match** (first action RF femur +0.42 vs +0.36, LB tibia −0.37 vs −0.35), so foot friction / contact is not
  the cause. The observation pipeline was already ruled out on 11-06 (replay matches to 7.7e-7, again 9.5e-7 today). IMU noise is in
  the training range (sim noise acc 0.05 m/s², gyro 0.02 rad/s). Freezing the IMU inputs offline removes only ~25 % of the 3 Hz dither.
- After onset the robot settles into a slow sway (0.1–1 Hz, middle coxas ~5° peak-to-peak) plus sub-tick dither. The left-back femur
  and tibia sit 0.08 rad behind their command (foot pressed into the floor, about half PWM).

### Why the policy faults (22 % dropped reads)
- The burst steps (Δq_cmd 0.05–0.2 rad) make one servo stop answering the SYNC_READ, most often LF tibia (sim idx 15), sometimes idx 8,
  9 or 13. Every later id in the read waits for it and drops too. The bytes received are always an exact multiple of 15 (clean replies,
  then silence): no corrupted packets and no hardware alert bits. The 3 ms retry of the same ids usually also fails. It never
  happens in STAND or SIT.
- Voltage: 10.8 V in STAND (11.4–11.6 V on 11-06, so the battery was low), one 9.8 V reading on the first burst step, otherwise
  10.5–10.9 V. Supply sag is not the main cause. Capping the power (run B) cut the onset drops from ~50 to 4, so the current spikes
  are what triggers it.
- New log fields: `hv` (voltage of this step's health read, nan if none) and `rd` (bytes and replies of the first read, ids recovered
  by the retry, alert count).

### Command shaping (opt-in flags; defaults = the sim-faithful path)
`--soft-start S` (blend the policy target from STAND over S s), `--rate-limit R` (rad/s cap per joint on q_cmd), `--policy-pwm N`
(Goal PWM under the policy, 885 = 100 %). The obs gets the shaped (= sent) q_cmd. `--imu-calib-bias`: if the start-up stillness
check fails (hanging robot), keep the calibrated gyro bias.

| hung run | onset (1 s) | standing cmd 0 | walking v_x 0.05 |
|---|---|---|---|
| none | Δq_cmd ≤ 0.13 rad/step, 38–63 drops, fault in 0.5 s | — | — |
| **A** `--soft-start 1.0 --rate-limit 3.5` | ≤ 0.023 rad/step, **0 drops**, gyro std 0.02 | calm, 0–3 drops | drops on 10/110 (fault) and 42/884 steps |
| **B** `--policy-pwm 619` | ≤ 0.09 rad/step, 4 drops, jitter still felt | calm, 0 drops | Δq_cmd ≤ 0.21 rad/step, drops on 14/744 steps, no fault |

- User: in A the stand → policy hand-off was smooth with no jitter. Walking at 0.05 felt fast: the legs moved smoothly but suddenly,
  and the harness swung from the inertia. In B there was jitter at policy on and at stop, and the gait felt slightly jittery at
  times (the 50 Hz log is calm in the 1 s after stop).
- While walking the measured joints move faster than the 0.07 rad/step cap (|Δq| p99 0.09–0.10 rad/step): the XL430 overshoots with
  D 0. The policy steps at 1.5 Hz with fast swings in sim too. Hung walking is not representative: the legs are unloaded and the
  kinematic velocity estimate has no ground contact.
- Speeds below 0.05 m/s (the trained minimum) in sim: cmd 0.025 → 0.021 ± 0.010 m/s achieved (0.05 → 0.047), same 1.5 Hz step rate,
  shorter strides, swing just as fast.

### Next
- Robot: floor test with a charged battery, A shaping + `--policy-pwm 619`, starting at v_x 0.02 with finer speed steps.
- hexasim (the real fix): a stronger action-rate / smoothness penalty, and the XL430 actuator model so the chatter costs something in
  training (the servo bench from the 2026-11-11 task list feeds it).

## 2026-11-11 (later): dropped reads are a broken SYNC_READ chain; read order + loop timing fixes (offline, not yet on the robot)

Analysis of all 11-06 / 11-11 logs (`rd` field): in **every** step with a final drop, the missing set is a suffix of the read
list (sim order). The first missing id (the culprit) is a servo that moves fast at that moment (its own |Δq_cmd| 0.05 rad vs
median 0.002). It keeps tracking through its drops, so it is not rebooting. The culprits were LF tibia (id 9) most of the time,
then RM/RB tibia and LF/RB/LM femur. The immediate retry rescued 109 of 165 incomplete reads in run A. When it fails it
usually gets nothing, not even from the ids behind the culprit that it asks first.

Changes:
- `dxl_bus.read_positions`: **adaptive read order**. Each servo that breaks the chain moves towards the end of the SYNC_READ
  list (decayed counts). A break then costs the culprit alone. On the logs: run A 212 → ~56 dropped reads, joints that ever
  drop 11 → the 4 culprits. The culprit's own drops are unchanged (that is the current-spike problem: shaping, battery).
  `rd` now has 5 fields (… , culprit id or 0), and the runner summary lists the chain breakers.
- Loop timing: policy steps were compute p50 16.8 / **p99 20.9 ms** (over the 20 ms period), and 19.6–23 ms on every
  retry step. Now the IMU read runs in a worker thread overlapped with the SYNC_READ: measured on the Pi −1.65 ms per step
  (bus-wait stand-in 5.3 ms + IMU: 7.20 → 5.55 ms p50). `--no-imu-thread` turns it off. The health read is skipped on steps
  that are already 16 ms in.
- `tools/bus/chain_probe.py` (read-only, torque off, **not run yet, servo power was off**): reproduces the broken chain
  with a ghost id, measures how long the ids behind it stay deaf (SYNC_READ vs single READ retry), and tests FAST SYNC READ
  (0x8A, X-series firmware ≥ 45; ours is 45) for time and ghost behaviour. `bus_bench.py --imu-thread` times the overlap.

Higher read rate: it does not help. The policy samples q once per 20 ms (q̇ = 2-step difference, W = 50 window) and a 50 Hz
checkpoint must run at 50 Hz. Extra reads add no information to the observation, only bus time and exposure. The wins are a
shorter read (FAST SYNC READ, if the probe confirms) and a cheaper retry. Optional: I2C runs at 100 kHz (`clock-frequency`
0x186a0). 400 kHz (`dtparam=i2c_arm_baudrate=400000`, reboot) would make the IMU read ~0.5 ms; the thread already hides it.

## 2026-11-12: Floor teleop of 2026-11-11 13:01–13:05, diagnosis (logs only, no robot run)

Logs `~/hexapod_logs/2026-11-11_13-01-51`, `_13-03-14`, `_13-05-01`. Flags: `--soft-start 1.0 --rate-limit 3.5 --policy-pwm 619`.
Speed steps 0.02–0.05. Analysis script: per-state means of `q`, `q_cmd`, FK of the toes (`leg_ik.LegKin`).

- **The only real overload is the first one.** Session 1, policy at zero command: within 1.5 s the LB femur (id 2) is commanded
  +0.868 rad and stops at +0.747 (error 0.121 rad), and the LB tibia (id 3) stops 0.104 rad short. Both stay pinned for ≈ 4 s, then
  id 2 reports `overload` (5.7 s after policy on). At P 1061 a 0.121 rad error asks for 8.29 PWM/pulse × 79 pulses = 655, above the
  619 Goal PWM cap, so the femur sat at its PWM limit against a planted foot until the protection tripped. In the sim the same
  checkpoint runs at 885 (it would ask 74 %, unsaturated), and the sim has no overload protection at all.
- **The later faults are the same alarm, still latched.** The XL430 overload bit (HW error 0x20) latches, turns that servo's torque off,
  and is cleared only by a Reboot instruction or a power cycle. The sit-down after fault 1 ran with id 2 limp, and the LB tibia (id 3)
  then lost tracking by 0.94 rad (overloaded carrying the leg alone). The next `u` (same session) and the next session (13-03-14)
  faulted 10 ms after torque-on with "id 2 overload; id 3 overload", before anything moved. `setup_servos` does not read HW_ERROR and
  never reboots, so a latched alarm survives a runner restart. Session 3 (power cycled in between) stood up fine and stopped on a
  10.4 V battery fault 7 s into the policy (STAND 10.8–11.0 V: the battery was low).
- **The policy's standing pose is less stable than STAND, measurably.** Toe positions by FK (last 2 s of each state):

  | | toe r from the coxa axis (cm) RF RM RB LF LM LB | toe z (cm) | middle toes y | front/back toes x |
  |---|---|---|---|---|
  | STAND (q_default) | 6.5 6.7 6.7 6.5 6.4 6.6 | −16.9 … −17.0 | ±16.5 | ±17.1 |
  | policy, cmd 0 | 3.5 0.6 1.4 0.6 0.1 4.9 | −16.4 … −17.7 | −10.5 / +10.1 | +15.0 / +12.9, −13.4 / −16.1 |

  The policy pulls every toe almost under its hip, so the support polygon shrinks (middle feet 6 cm further in). Its six commanded
  toes are not coplanar: against the best-fit plane, LB is 2.6 mm and RF 1.6 mm low, while the middle legs are 2.2 / 3.6 mm high
  (`tools/bench/policy_stance_2026-11-11.json`, measured: LB −6.1, RF −4.9 mm). So the body rests on the RF–LB diagonal, rocks, and
  LB pushes against the rest (the stalled femur/tibia above). In sim this checkpoint already stands badly (hexasim §2.4.1: standing
  reward 0.53 vs 1.05 for zero action).

### Runner changes this needs (not made yet)
- At start-up read HW_ERROR on all 18. If an alarm is latched, report it and offer a Reboot (instruction 0x08), then re-apply the RAM
  gains (a reboot resets them).
- A stall monitor: log Present PWM (addr 124) in the health read and WARN → freeze/sit when a joint sits at its Goal PWM cap for
  > ~1 s, well before the ≈ 4 s it took to trip.
- After an overload fault, sit down with that leg's other joints at reduced PWM, or just torque off (the limp leg overloads the next).

## 2026-11-12: Servo bench + ground test code for the XL430 motor model (written, offline-tested; not run on the robot yet)

User constraints: no scale or known weight (a sealed drink/food package with its net mass on the label works for the optional
torque calibration), one leg cannot come off, hung-up tests possible, 15 cm ruler, no bench supply (battery, charged). Allowed:
switching one servo to PWM mode temporarily, deliberately tripping the overload protection. Battery-sensitive tests run first.

Code: `tools/bench/` — `servo_bench.py` (CLI, tests), `bench_lib.py` (servo layer on `dxl_bus`, recorder, safety, `SimServos`),
`bench_analysis.py` (per-run summary), `analyze_bench.py` (all runs → `bench_summary.json`, PWM→torque calibration),
`test_bench_offline.py` (16 tests on a simulated XL430), `policy_stance_2026-11-11.json` (the stance the policy commanded).
Logs: `~/hexapod_logs/bench/<stamp>_<test>.{npz,json}` — raw samples (one servo at ~0.7–1 kHz, or all 18 at 50 Hz) for hexasim to
replay and fit, plus the test definition.

Beyond the runner (which writes RAM only), the bench (a) writes Operating Mode (EEPROM 11) of one servo 3 ↔ 16 with its torque off,
recorded first in `~/hexapod_logs/bench/.pwm_mode_ids.json` and restored at every bench start (the teleop refuses mode ≠ 3 anyway);
(b) sends REBOOT (0x08) to clear latched alarms, then re-applies P 1061 / I 0 / D 0 / no profile. It reads signed PWM/load/velocity,
so it parses byte-stuffed status packets (`parse_status_any`; the runner's position-only reads never stuff).
Aborts: any hardware alert, servo ≥ 65 °C, any voltage < 10.0 V, leaving the motion window; then safe stop (PWM-mode servo back to
mode 3, torque off).

| order | command | setup | gives (xl430.py) |
|---|---|---|---|
| 0 | `servo_bench.py status` | none, torque stays off | battery V, temps, latched alarms (offers reboot), modes |
| 1 | `servo_bench.py hung` | robot hung, feet clear ≥ 25 cm | in order: **speed** (RF coxa, PWM mode, duty 0.05–1.0, ±0.45 rad window, plug-brake at the edge: ω_nl(V), Coulomb duty, t63) → **step** (RF coxa/femur/tibia, 0.02–0.4 rad, caps 885 and 619: dead time, rise, overshoot, final error) → **sine** (same joints, ±0.05 rad at 0.5–25 Hz, 50 Hz ZOH: gain, lag vs hexasim §2.4.2) → **hold** (RF femur at 7 angles from both sides: leg gravity moment, friction, P-law check) → **drops** (all 18 zig-zag |Δq_cmd| 0–0.2 rad/step: drop rate vs step size) |
| 2 | `servo_bench.py hold --mass M --mass-label "..."` (optional) | hung, the package hung from the RF toe (string loop) | PWM→torque: τ_stall(V) = M g arm / Δduty vs the run without the mass (refuses > 0.75 N m) |
| 3 | `servo_bench.py stall --joint LB_femur` | belly on the floor, body held down / weighed | contact search (toe down into the floor, 30 %), then cap:request trials 30:45, 50:75, 70:100, 100:60, 100:130 % up to 40 s: time to the overload alarm; reboot + cool-down between trials |
| 4 | `servo_bench.py ground` | belly on a flat floor, 40 cm clear | stand-up, STAND 10 s, step one leg at a time into the 2026-11-11 policy stance, hold at Goal PWM 619 and 885, back, lift each tripod 2 cm for 3 s, sit; per-joint static PWM, voltage under load; stall monitor (|PWM| ≥ 92 % of the cap for 0.5 s) ends a hold before the ~4 s overload; tilt > 25° aborts |

Afterwards: `python3 tools/bench/analyze_bench.py` (Pi or workstation after rsync of `~/hexapod_logs/bench`).
Offline: `python3 tools/bench/test_bench_offline.py` — 16/16 on the workstation; sim checks include ω_nl / Coulomb recovered by the
speed fit, τ_stall recovered by the mass calibration within 0.2 N m, stall trip timing and reboot, the PWM-mode restore after a crash,
a latched alarm blocking a start, stuffed packets, the stand-up reversing after a mid-stand-up alarm, the stance leg steps keeping
the moving toe ≥ 1.5 cm up.
Known limits: no torque reference without a known mass (then τ_stall stays the datasheet's, and only the duty-unit numbers are
measured); backlash is invisible to the output-shaft encoder (needs an external reference); hung drops are unloaded legs (the ground
test logs drops under load); foot friction is not measured.

## 2026-11-13: XL430 bench run on the robot (hung, mass, stall, ground)

All four bench stages ran; logs are in `~/hexapod_logs/bench/` (Pi and workstation), summary in `bench_summary.json`. Details,
log stems and open issues are in `handoff.md`.

| quantity | measured | conditions |
|---|---|---|
| no-load speed | 5.35 rad/s (51 rpm) | RF coxa, PWM mode, ~11.7 V |
| velocity time constant t63 | ~75 ms | duty ≥ 30 % |
| friction | static 8–12 % duty, Coulomb ~1.8 % | RF coxa |
| stall torque | 1.33 N·m (fit r² 0.93) | RF femur, 0.465 kg at the toe, 11.2 V |
| step response | dead time 13–40 ms, ~20 % overshoot at 0.1–0.2 rad, final error 0.5–7 mrad | RF leg, caps 619 and 885 |
| frequency response | flat to ~3 Hz, peak ~1.25 near 5 Hz, gone by 12.5 Hz, ~50 ms lag | ±0.05 rad, 50 Hz ZOH |
| P-law check | measured/predicted 1.014 | hold, no mass |
| sag under load | 60 mrad at 0.44 N·m | hold with mass |
| overload trip | none at 30 % (40 s); 15.2 s at 50 %; 7.7 s at 56 %; ~4.2 s from 63 % to 100 % | LB femur stalled on the floor, 39–43 C |
| read drops (hung) | 0 up to 0.10 rad steps at cap 619; 0.24 % at 0.15 rad; start when V ≤ ~10.5 V; culprits ids 9, 7, 18 | all 18 zig-zag |
| standing load | STAND: tibia ~14 %, femur ~4 %, coxa ~1.5 % duty; peak 35 % during stand-up | ground test, 11.3 V |
| policy stance load | LB tibia 27 %, LB femur 21 %, others ≤ 12 %; same at caps 619 and 885 | 2026-11-11 stance |
| ground test totals | no stall, 0 drops, tilt max 7.2°, V min 10.9 | |

Battery: 11.5 V at rest fell below 10 V with all 18 servos reversing at the 100 % cap, and one stalled femur at 100 % pulled an
11.6 V pack to 10.7 V. The default stall requests did not saturate the 70 % and 100 % caps (the foot gives 90–130 mrad), so the
flat ~4.2 s figures come from a rerun with `--trials 0.7:1.6,0.85:1.8,1.0:2.0` and more weight on the body.

Runner changes the same day (offline-tested, not yet run on the robot): `setup_servos` refuses to start on a latched hardware alarm;
`Safety` has a stall monitor on the duty estimated from the P law (`min(|q_cmd − q| / 0.164, cap)`, no bus read): FAULT at ≥ 60 % for
1.0 s or ≥ 45 % for 5.0 s. Replayed on `2026-11-11_13-01-51` it faults 2.8 s before the real id 2 overload; on the ground log the
estimate matches the measured PWM (corr 0.997) and nothing trips.

Next: build hexasim's XL430 actuator model from these logs; reseat the LF connectors.

## 2026-11-13 (evening): S1 XLS exported and on the Pi; test protocol (nothing run on the servos yet)

Decision (user, hexasim session): test the XL430-trained policy on the robot **before** the motor model is refitted or anything
is retrained; speed from phone video. Checkpoint: hexasim `logs/rsl_rl/hexapod_s1/2026-11-12_09-36-47_s1_xls_seed42/model_7999.pt`
(`Hexapod-S1-XL430S-v0`, blind, trained on the provisional XL430 model with a 3.5 rad/s command slew limit).

### What changed
- **Policies are picked by tag.** `tools/policy/export_contract.py <ckpt> <tag>` writes `config/contract_<tag>.json` and
  `models/<tag>/` (tag `s1h4`, the default, keeps `models/`). The runner takes `--policy <tag>`; tools and tests read
  `HEXAPOD_POLICY` (default `s1h4`). `export_fused.py` reads the actor width from the checkpoint (1,125 for S1).
- **Servo tick from the contract.** `ProprioReader(n, dt, tick)`: AX-12A grid for the AX-trained S1H4, 2π/4096 for S1 XLS.
- **Gait clock** (`s1h_policy.Clock`). S1 keeps the clock (S1H4 dropped it, so the runner had none): f = clamp(v_x / 0.20, 0.8, 2.5)
  Hz, tripod offsets, all zeros under a zero command; the command of the step that just ended decides, as in the env.
- **Slew limit from the contract** (`rate_limit` 3.5 rad/s = 0.07 rad per step, `s1h_policy.slew`): on by default for this
  policy because it trained with it. `--rate-limit` still overrides; `--soft-start` and `--policy-pwm` stay deploy-side extras.

### Checks (all offline)
- Pi: `HEXAPOD_POLICY=s1xls python3 tools/test_teleop_offline.py` 38 / 38; parity with hexasim's modules, action error 7.9e-6
  over 120 steps (o_p 7.9e-6). `s1h4` unchanged: 38 / 38, 1.2e-6.
- Clock and slew against the real env, not a re-implementation: `tools/sim/sim_clock_slew_record.py` (hexasim, 32 envs × 12 s,
  commands resampled every 2–3 s) → `tools/sim/check_clock_slew.py`: clock error 2.2e-5 over 19,168 steps (138 command changes,
  2,061 standing steps), sent target 7.4e-8. The slew limit is active on 71 % of the sim steps.
- hexasim replayed the hung coxa step / sine logs through its XL430 law: the model the policy trained on reproduces them
  (gain within ≈ 0.1 and lag within 5 ms from 1.5 to 25 Hz; hexasim phase2-progress §2.4.4).
- Not checked: `tools/bench/test_bench_offline.py` did not finish within 2 min on the workstation today (a copy started by an
  earlier session is still running). Not looked into.

### Protocol (robot, user present)
Battery charged (≥ 11.8 V at rest), LF connectors reseated. Start every session with
`python3 tools/bench/servo_bench.py status`. Runner: `python3 scripts/hexapod_teleop.py --policy s1xls --soft-start 1.0`
(add `--imu-calib-bias` when hung, `--vx-max 0.15` for the 0.15 m/s runs). Goal PWM under the policy stays 885 (as trained);
fall back to `--policy-pwm 619` only if reads drop.

1. **Hung**, feet clear: `u`, `p`, zero command 20 s, then `w` to 0.05 / 0.075 / 0.10 for 10 s each, `space`, `p`, `j`.
   Pass: no fault, legs step as two tripods at ≈ 0.8 Hz, no drops at onset.
2. **Floor, zero command** 30 s. Note vibration, stall warnings, the stance.
3. **Floor, walking**: v_x 0.05, 0.10, 0.15; ≥ 3 runs of ≥ 20 s each, tape marks every 0.5 m, phone video from the side.
4. **Turn on the spot**: `q` / `e` at 0.2 rad/s, one full turn each way.
5. Optional, only if 3 walks: a ≈ 3 cm board at 0.10 m/s.

Per run: log stem (`~/hexapod_logs/<stamp>`), video file, falls / faults, anything seen. From the logs afterwards: speed against
video, `q_cmd − q` by joint type in stance and swing, estimated duty, dropped reads and culprits, loop p99, voltage, Σ Δa² —
each beside the sim's number for the same checkpoint.

## 2026-11-13 (night): first S1 XLS runs on the robot (user at the robot), log analysis

Runner: `--policy s1xls --soft-start 1.0`, slew 3.5 rad/s, Goal PWM 885. Logs `~/hexapod_logs/` (Pi clock): `2026-11-13_14-25-40`
(hung by the IMU: tilt 7°, 13.5 s at zero command, 6 s at 0.05), `_14-27-13` (floor: 52 s zero command, 14.7 s at 0.05, 2.7 s at
0.075), `_14-30-37` (floor: 20 s at 0.05; larger joint travel and body acceleration, taken to be the run over the ≈ 5 cm objects —
not confirmed against the video). **No fault in any session, dropped reads ≤ 0.01 %, loop period p99 20.0 ms (compute p99 17 ms).**
User: walked on the flat floor; crossed the objects slowly without getting stuck. Speed not measured yet.

**What the logs hold: walking at 0.05 m/s only** (plus 2.7 s at 0.075); no 0.10 / 0.15 and no turn commands.

Same checkpoint in sim (`hexasim logs/eval/xl430/chatter_s1_xls_seed42_v{0.0,0.05}.npz`, 64 envs, flat) against the floor log
`_14-27-13`, coxa / femur / tibia:

| at 0.05 m/s | sim | robot |
|---|---|---|
| step rate (femur spectrum) | 0.80 Hz | 0.77 Hz |
| joint travel, peak to peak (rad) | 0.49 / 0.75 / 0.77 | 0.48 / 0.81 / 0.89 |
| tracking error `q_cmd − q`, RMS (mrad) | 42 / 77 / 77 | 61 / 113 / 116 |
| share of steps with error > 60 % of the saturation error | 0.03 / 0.21 / 0.24 | 0.10 / 0.36 / 0.38 |
| joint speed p99 (rad/s) | 3.1 / 3.2 / 3.2 | 3.4 / 4.1 / 4.4 |
| Σ Δa² per step | 0.113 | 0.101 |
| standing, Σ Δa² per step | 0.0085 | 0.0004 |

- The gait transfers: same step rate and joint travel, no stall fault, standing calmer than in sim.
- **The real servos track about 45–50 % worse under load and then move faster than the 3.5 rad/s command slew** (catch-up). The
  sim's model does neither. This, not the unloaded response, is the gap the model fit has to close.
- The duty in these logs is the runner's estimate from the position error, not Present PWM; it reads 1.00 at p99 during swing,
  where back-EMF takes most of it. Present PWM was not logged while walking.
- Voltage 10.4–10.9 V under the policy (battery not full); chain breaks: RM tibia (id 15) 15, RB tibia (id 18) 8, LM tibia 3 over
  the three sessions, all recovered by the retry.
- Leg odometry from the logs (the policy's kinematic velocity, stance feet assumed not to slip): 0.043 m/s at 0.05 commanded in
  both floor sessions, 0.066 at 0.075. An upper estimate until the video is measured. The accelerometer cannot give speed
  (a 0.05 m/s² bias is 0.5 m/s after 10 s).

Missing for the model: walking at 0.10–0.20 m/s and turning, Present PWM while walking, total robot mass, measured speed.

