#!/usr/bin/env python3
"""Runner-side IMU block: MPU-6050 -> the 9 IMU dims of hexasim's o_p, in the sim's convention. NumPy only, Python 3.8.

Output of ImuBlock.update() = [accel (3), gyro (3), projected gravity (3)], all in the base_link frame
(+x forward, +y left, +z up, origin at the footprint centre in the coxa plane):
  accel   m/s^2, specific force at the base_link origin, gravity INCLUDED -> (0, 0, +9.81) when level
          (Isaac Lab ImuCfg adds a -g bias; the real sensor is moved to the origin by subtracting alpha x r + w x (w x r))
  gyro    rad/s, bias-corrected
  gravity unit vector, (0, 0, -1) when level (hexasim projected_gravity_b)
Calibration comes from config/imu_calib.json (written by tools/imu/imu_calib.py solve).

    blk = ImuBlock("config/imu_calib.json"); imu = MPU6050()
    blk.start([imu.read() for _ in range(100)])        # ~2 s, robot still: re-measures gyro bias + initial gravity
    x9 = blk.update(*imu.read(), t=time.monotonic())   # every control step

The MPU6050 register setup mirrors tools/imu/imu_calib.py (DLPF ~44 Hz, +-8 g, +-500 deg/s): change both together.
"""
import fcntl
import json
import os
import struct
import time

import numpy as np

G0 = 9.80665
ACC_LSB_PER_G, GYR_LSB_PER_DPS = 4096.0, 65.5  # +-8 g, +-500 dps


class MPU6050:
    def __init__(self, bus="/dev/i2c-1", addr=0x68):
        self.fd = os.open(bus, os.O_RDWR)
        fcntl.ioctl(self.fd, 0x0703, addr)  # I2C_SLAVE
        assert self._rd(0x75, 1)[0] == 0x68, "WHO_AM_I mismatch"
        for reg, val in ((0x6B, 0x01), (0x1A, 0x03), (0x19, 0x00), (0x1B, 0x08), (0x1C, 0x10)):
            os.write(self.fd, bytes([reg, val]))
        time.sleep(0.1)

    def _rd(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, n)

    def read(self):
        """-> (accel m/s^2, gyro rad/s) in the raw sensor frame."""
        ax, ay, az, _t, gx, gy, gz = struct.unpack(">7h", self._rd(0x3B, 14))
        return (np.array([ax, ay, az]) / ACC_LSB_PER_G * G0,
                np.radians(np.array([gx, gy, gz]) / GYR_LSB_PER_DPS))


def _cross(a, b):
    """3-vector cross product (np.cross costs ~60 us per call on the Pi 4; this ~5 us)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _unit(v):
    return v / np.linalg.norm(v)


class ImuBlock:
    def __init__(self, calib_path, tilt_gain=0.02, g_sigma=0.7):
        """tilt_gain: accel correction per step at rest (0.02 at 50 Hz = ~1 s time constant).
        g_sigma: m/s^2; the accel correction fades as | |a| - g | grows past this (body is accelerating)."""
        c = json.load(open(calib_path))
        self.R = np.array(c["R_body_from_sensor"])  # v_body = R @ v_sensor
        self.off, self.sc = np.array(c["acc_offset"]), np.array(c["acc_scale"])
        self.gyro_bias = np.array(c["gyro_bias_sensor"])  # sensor frame
        self.r = np.array(c.get("lever_arm_body_m", [0.0, 0.0, 0.0]))
        self.tilt_gain, self.g_sigma = tilt_gain, g_sigma
        self.reset()

    def reset(self):
        self.u = None  # unit "up" vector in the body frame (= -projected gravity)
        self.alpha = np.zeros(3)
        self.w_prev = self.t_prev = None
        self.ab = np.zeros(3)  # last calibrated body-frame accel at the sensor (no lever-arm correction)

    def body(self, a_raw, w_raw):
        return self.R @ ((a_raw - self.off) * self.sc), self.R @ (w_raw - self.gyro_bias)

    def start(self, samples, max_bias_change=0.1):
        """samples: [(a_raw, w_raw), ...] taken with the robot still. Re-measures the gyro bias (it drifts with
        temperature) and seeds the gravity estimate. Raises ValueError (state unchanged) if the robot was not still."""
        A, W = np.array([s[0] for s in samples]), np.array([s[1] for s in samples])
        if W.std(0).max() > 0.02 or A.std(0).max() > 0.15:
            raise ValueError("IMU not still during start-up bias capture (gyro std %s)" % np.round(W.std(0), 4))
        new = W.mean(0)
        if np.abs(new - self.gyro_bias).max() > max_bias_change:
            raise ValueError("start-up gyro bias %s is far from calibrated %s: robot rotating or calibration stale"
                             % (np.round(new, 3), np.round(self.gyro_bias, 3)))
        self.gyro_bias = new
        self.reset()
        self.u = _unit(self.R @ ((A.mean(0) - self.off) * self.sc))

    def start_calibrated(self, samples):
        """Fallback when the robot cannot be held still (e.g. hanging): keep the calibrated gyro bias, seed gravity
        from the mean accel. Bias error = drift since calibration (temperature); for bench tests only."""
        A = np.array([s[0] for s in samples])
        self.reset()
        self.u = _unit(self.R @ ((A.mean(0) - self.off) * self.sc))

    def update(self, a_raw, w_raw, t):
        """One control step. t = monotonic seconds. Returns the 9-vector [accel@origin, gyro, projected gravity]."""
        ab, wb = self.body(a_raw, w_raw)
        self.ab = ab
        dt = 0.0 if self.t_prev is None else min(max(t - self.t_prev, 1e-3), 0.1)
        if self.w_prev is not None:  # angular acceleration: backward difference, EMA-smoothed
            self.alpha = 0.8 * self.alpha + 0.2 * (wb - self.w_prev) / dt
        self.w_prev, self.t_prev = wb, t
        # the sensor sits at r: it measures a_origin + alpha x r + w x (w x r); refer it to the origin as the sim does
        a_o = ab - _cross(self.alpha, self.r) - _cross(wb, _cross(wb, self.r))
        if self.u is None:
            self.u = _unit(ab)
        # a vector fixed in the world, seen from the rotating body: du/dt = -w x u
        self.u = _unit(self.u - _cross(wb, self.u) * dt)
        beta = self.tilt_gain / (1.0 + ((np.linalg.norm(a_o) - G0) / self.g_sigma) ** 2)
        self.u = _unit((1.0 - beta) * self.u + beta * _unit(a_o))
        return np.concatenate([a_o, wb, -self.u])
