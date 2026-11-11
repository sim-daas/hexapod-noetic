#!/usr/bin/env python3
"""Capture the robot's current joint angles as the runner's SLEEP pose (Pi, Python 3.8). Torque is not touched.

    python3 tools/poses/capture_sleep.py            # robot lying on its belly, legs folded, torque off

Reads every servo 25 times (0.5 s), takes the median per joint, and writes config/sleep_pose.json by joint name.
joint_map.Contract uses that file instead of the vendor sleep_positions when it exists. Delete it to go back.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import dxl_bus as dx  # noqa: E402
from joint_map import Contract, SLEEP_FILE  # noqa: E402


def main():
    c = Contract()
    bus = dx.DxlBus()
    ids = [int(i) for i in c.ids]
    try:
        reads = []
        for _ in range(25):
            q = bus.read_positions(ids)
            if all(v is not None for v in q):
                reads.append(q)
            time.sleep(0.02)
        if len(reads) < 10:
            raise SystemExit("too many dropped reads (%d/25 complete)" % len(reads))
        R = np.array(reads)
        q = np.median(R, axis=0)
        if (R.max(0) - R.min(0)).max() > 0.02:
            raise SystemExit("robot moving while capturing (spread %.3f rad)" % (R.max(0) - R.min(0)).max())
        out = np.clip(q, c.lo, c.hi)
        if np.abs(out - q).max() > 1e-9:
            print("note: clipped to the joint limits by up to %.3f rad" % np.abs(out - q).max())
        data = {"captured": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "relaxed belly-down pose with torque off, read from the servos",
                "q": {n: round(float(v), 4) for n, v in zip(c.names, out)}}
        json.dump(data, open(SLEEP_FILE, "w"), indent=1)
        for i, n in enumerate(c.names):
            print("%-20s %+.3f rad (vendor %+.3f)" % (n, out[i], c.raw["sleep_positions"][i]))
        print("wrote", SLEEP_FILE)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
