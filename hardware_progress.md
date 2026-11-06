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
