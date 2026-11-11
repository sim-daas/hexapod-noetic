#!/usr/bin/env python3
"""Pi side of the motion check: run the runner's ImuBlock and print one JSON line per sample on stdout.

  python3 tools/imu/imu_stream.py [--calib config/imu_calib.json] [--rate 50] [--secs 0]

Keep the robot STILL for the first ~2 s: the gyro bias is re-measured then (same step the runner does at start-up).
Line = {"t": monotonic s, "ab": calibrated body accel at the sensor, "ao": accel referred to the origin,
        "w": body gyro, "g": projected gravity}. Status messages go to stderr. Needs only numpy + /dev/i2c-1.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from imu_block import ImuBlock, MPU6050  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default=os.path.join(HERE, "..", "..", "config", "imu_calib.json"))
ap.add_argument("--rate", type=float, default=50.0)
ap.add_argument("--secs", type=float, default=0.0, help="stop after this many seconds (0 = until killed)")
a = ap.parse_args()

imu, blk = MPU6050(), ImuBlock(a.calib)
print("imu_stream: hold the robot still for 2 s (gyro bias)...", file=sys.stderr, flush=True)
try:
    blk.start([(imu.read(), time.sleep(0.02))[0] for _ in range(100)])
    print("imu_stream: start-up bias %s rad/s (sensor frame). Go ahead and move it." % [round(x, 4) for x in blk.gyro_bias],
          file=sys.stderr, flush=True)
except ValueError as e:
    print("imu_stream: WARNING %s -> using calibration-file bias" % e, file=sys.stderr, flush=True)

period, t0 = 1.0 / a.rate, time.monotonic()
nxt = t0
while a.secs <= 0 or time.monotonic() - t0 < a.secs:
    ar, wr = imu.read()
    t = time.monotonic()
    x = blk.update(ar, wr, t)
    print(json.dumps({"t": round(t, 4), "ab": [round(v, 4) for v in blk.ab], "ao": [round(v, 4) for v in x[0:3]],
                      "w": [round(v, 5) for v in x[3:6]], "g": [round(v, 5) for v in x[6:9]]}, separators=(",", ":")),
          flush=True)
    nxt += period
    time.sleep(max(0.0, nxt - time.monotonic()))
