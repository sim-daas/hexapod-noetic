#!/usr/bin/env python3
"""MPU-6050 (GY-521) driver + calibration for the PhantomX Mark IV. Runs on the Pi: python3 3.8, numpy, no smbus.

Target convention = hexasim Isaac Lab ImuCfg on base_link (URDF pxmark4: +x forward, +y left, +z up, origin at the
body-footprint centre):
  accel  = specific force in m/s^2, gravity INCLUDED -> (0, 0, +9.81) when level (Isaac Lab adds a -g bias)
  gyro   = rad/s, body frame
  grav_b = unit gravity direction in body frame -> (0, 0, -1) when level (projected_gravity_b)

Commands (poses are held still for SECS seconds; files go to ~/imu_calib/):
  capture NAME [SECS]   record raw sensor-frame samples (NAME = level | nose_up | left_up | px py pz nx ny nz ...)
  six                   interactive six-face capture of the bare board (auto-detects each face)
  solve [rx ry rz]      (rx..rz = IMU position from the base_link origin in metres; stored as lever_arm_body_m)
                        six-face (if present) -> accel offset/scale; level+nose_up(+left_up) -> R_body<-sensor;
                        gyro bias from level. Writes ~/imu_calib/calib.json
  live [SECS]           print calibrated body-frame accel / gyro / gravity at ~10 Hz (needs calib.json)
"""
import fcntl, json, os, struct, sys, time
import numpy as np

ADDR, BUS = 0x68, "/dev/i2c-1"
G0 = 9.80665
ACC_LSB_PER_G, GYR_LSB_PER_DPS = 4096.0, 65.5  # +-8 g, +-500 dps
DIR = os.path.expanduser("~/imu_calib")


class MPU6050:
    def __init__(self):
        self.fd = os.open(BUS, os.O_RDWR)
        fcntl.ioctl(self.fd, 0x0703, ADDR)  # I2C_SLAVE
        assert self._rd(0x75, 1)[0] == 0x68, "WHO_AM_I mismatch"
        for reg, val in ((0x6B, 0x01),   # PWR_MGMT_1: PLL on X gyro
                         (0x1A, 0x03),   # CONFIG: DLPF ~44 Hz
                         (0x19, 0x00),   # SMPLRT_DIV: 1 kHz internal
                         (0x1B, 0x08),   # gyro +-500 dps
                         (0x1C, 0x10)):  # accel +-8 g
            os.write(self.fd, bytes([reg, val]))
        time.sleep(0.1)

    def _rd(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, n)

    def read(self):
        """-> (accel m/s^2 (3), gyro rad/s (3)) in the raw sensor frame."""
        ax, ay, az, _t, gx, gy, gz = struct.unpack(">7h", self._rd(0x3B, 14))
        return (np.array([ax, ay, az]) / ACC_LSB_PER_G * G0,
                np.radians(np.array([gx, gy, gz]) / GYR_LSB_PER_DPS))


def unit(v):
    return v / np.linalg.norm(v)


def cmd_capture(name, secs=5.0):
    imu, A, W, T = MPU6050(), [], [], []
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        a, w = imu.read()
        A.append(a); W.append(w); T.append(time.monotonic())
        time.sleep(0.01)
    A, W, T = np.array(A), np.array(W), np.array(T)
    os.makedirs(DIR, exist_ok=True)
    np.savez(os.path.join(DIR, name + ".npz"), a=A, w=W, t=T)
    rate = (len(T) - 1) / (T[-1] - T[0])
    print("%s: n=%d rate=%.0f Hz | accel mean %s |a|=%.3f (std %s) | gyro mean %s (std %s)" % (
        name, len(T), rate, np.round(A.mean(0), 3), np.linalg.norm(A.mean(0)), np.round(A.std(0), 3),
        np.round(W.mean(0), 4), np.round(W.std(0), 4)))
    # Constant gyro offset is expected (that is the bias being measured); only scatter means motion.
    if W.std(0).max() > 0.02 or A.std(0).max() > 0.15:
        print("  WARNING: sensor was moving during capture (high scatter); redo this pose")


def load(name):
    p = os.path.join(DIR, name + ".npz")
    return np.load(p) if os.path.exists(p) else None


def cmd_six(timeout=600.0):
    """Interactive six-face capture. Face is auto-detected from which sensor axis carries +-1 g ('px' = +X axis up)."""
    imu, done, t_end = MPU6050(), {}, time.monotonic() + timeout
    names = {(0, 1): "px", (0, -1): "nx", (1, 1): "py", (1, -1): "ny", (2, 1): "pz", (2, -1): "nz"}
    os.makedirs(DIR, exist_ok=True)
    print("Rest the bare board still on one face at a time, in any order. Hold ~4 s each. Need 6 faces.", flush=True)
    last = None
    while len(done) < 6 and time.monotonic() < t_end:
        A, W, t0 = [], [], time.monotonic()
        while time.monotonic() - t0 < 3.0:
            a, w = imu.read(); A.append(a); W.append(w); time.sleep(0.01)
        A, W = np.array(A), np.array(W)
        m = A.mean(0)
        if W.std(0).max() > 0.02 or A.std(0).max() > 0.15:
            msg = "moving... hold it still"
        else:
            i = int(np.argmax(np.abs(m)))
            frac = abs(m[i]) / np.linalg.norm(m)
            name = names[(i, 1 if m[i] > 0 else -1)]
            if frac < 0.985:  # > ~10 deg off the axis
                msg = "still, but %s axis is %.0f deg off vertical; rest it flatter" % (name, np.degrees(np.arccos(frac)))
            elif name in done:
                msg = "%s already captured; turn to a different face (have %s)" % (name, sorted(done))
            else:
                np.savez(os.path.join(DIR, name + ".npz"), a=A, w=W, t=np.arange(len(A)))
                done[name] = m
                msg = "CAPTURED %s (%d/6)  mean %s  |a|=%.3f" % (name, len(done), np.round(m, 3), np.linalg.norm(m))
                last = None
        if msg != last:
            print(msg, flush=True)
            last = msg
    if len(done) < 6:
        print("timed out with %s; rerun to continue" % sorted(done)); return
    print("all six faces captured; run `solve` (or read the numbers below)")
    for i, ax in enumerate("xyz"):
        hi, lo = done["p" + ax][i], done["n" + ax][i]
        print("axis %s: reads %+.3f / %+.3f  -> offset %+.3f m/s^2, scale %.4f" % (ax, hi, lo, (hi + lo) / 2, 2 * G0 / (hi - lo)))


def cmd_solve(lever=None):
    """lever = IMU position in base_link (m, +x fwd, +y left, +z up from the base_link origin); kept from the old
    calib.json when not given."""
    out = {"acc_offset": [0.0] * 3, "acc_scale": [1.0] * 3}
    old = os.path.join(DIR, "calib.json")
    out["lever_arm_body_m"] = lever if lever is not None else (
        json.load(open(old)).get("lever_arm_body_m", [0.0] * 3) if os.path.exists(old) else [0.0] * 3)
    faces = {n: load(n) for n in ("px", "nx", "py", "ny", "pz", "nz")}
    if all(v is not None for v in faces.values()):
        for i, ax in enumerate("xyz"):
            hi, lo = faces["p" + ax]["a"].mean(0)[i], faces["n" + ax]["a"].mean(0)[i]
            out["acc_offset"][i] = (hi + lo) / 2
            out["acc_scale"][i] = 2 * G0 / (hi - lo)
        print("six-face: offset %s m/s^2, scale %s" % (np.round(out["acc_offset"], 3), np.round(out["acc_scale"], 4)))
    else:
        print("six-face captures absent: accel offset/scale left at 0 / 1")
    off, sc = np.array(out["acc_offset"]), np.array(out["acc_scale"])
    fix = lambda a: (a - off) * sc  # noqa: E731

    lv, nu, lu = load("level"), load("nose_up"), load("left_up")
    assert lv is not None and nu is not None, "need captures: level and nose_up"
    u0, u1 = unit(fix(lv["a"].mean(0))), unit(fix(nu["a"].mean(0)))
    z = u0
    d = u1 - u0
    d = d - d.dot(z) * z  # strip the z part, what is left is the body +x direction
    assert np.linalg.norm(d) > 0.1, "nose_up pose too close to level; tilt by >= ~15 deg"
    x = unit(d)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # v_body = R @ v_sensor
    print("pitch of nose_up pose: %.1f deg  (check against a protractor/known block height)" %
          np.degrees(np.arccos(np.clip(u0.dot(u1), -1, 1))))
    if lu is not None:
        u2 = R @ unit(fix(lu["a"].mean(0)))
        print("left_up check: body-frame up = %s  -> y>0 and x~0 expected (left side raised)" % np.round(u2, 3))
    out["R_body_from_sensor"] = R.tolist()
    out["gyro_bias_sensor"] = lv["w"].mean(0).tolist()
    print("R_body_from_sensor =\n%s" % np.round(R, 4))
    print("level pose in body frame: accel %s, gravity %s" % (
        np.round(R @ fix(lv["a"].mean(0)), 3), np.round(-(R @ fix(lv["a"].mean(0))) / G0, 3)))
    print("gyro bias (sensor frame) %s rad/s" % np.round(out["gyro_bias_sensor"], 4))
    json.dump(out, open(os.path.join(DIR, "calib.json"), "w"), indent=1)
    print("wrote", os.path.join(DIR, "calib.json"))


def cmd_live(secs=30.0):
    c = json.load(open(os.path.join(DIR, "calib.json")))
    R, off, sc = np.array(c["R_body_from_sensor"]), np.array(c["acc_offset"]), np.array(c["acc_scale"])
    gb, r = np.array(c["gyro_bias_sensor"]), np.array(c.get("lever_arm_body_m", [0.0] * 3))
    imu, t_end = MPU6050(), time.monotonic() + secs
    alpha, wb_prev, t_prev, k = np.zeros(3), None, None, 0
    while time.monotonic() < t_end:
        a, w = imu.read()
        t = time.monotonic()
        ab, wb = R @ ((a - off) * sc), R @ (w - gb)
        if wb_prev is not None:  # angular acceleration: backward difference, EMA-smoothed
            alpha = 0.8 * alpha + 0.2 * (wb - wb_prev) / (t - t_prev)
        wb_prev, t_prev = wb, t
        # accelerometer at r measures a_origin + alpha x r + w x (w x r)  ->  subtract to refer it to the origin
        a_o = ab - np.cross(alpha, r) - np.cross(wb, np.cross(wb, r))
        k += 1
        if k % 10 == 0:
            print("acc_b %s | acc@origin %s | gyro_b %s | grav_b %s" % (
                np.round(ab, 2), np.round(a_o, 2), np.round(wb, 3), np.round(-unit(ab), 3)))
        time.sleep(0.01)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "capture" and len(sys.argv) > 2:
        cmd_capture(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 5.0)
    elif c == "six":
        cmd_six()
    elif c == "solve":  # solve [rx ry rz]  (IMU position from base_link origin, metres)
        cmd_solve([float(v) for v in sys.argv[2:5]] if len(sys.argv) >= 5 else None)
    elif c == "live":
        cmd_live(float(sys.argv[2]) if len(sys.argv) > 2 else 30.0)
    else:
        print(__doc__)
