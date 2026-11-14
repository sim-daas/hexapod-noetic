#!/usr/bin/env python3
"""Summarise every bench log in a directory and collect the numbers hexasim's actuators/xl430.py needs.

    python3 tools/bench/analyze_bench.py [~/hexapod_logs/bench] [--out bench_summary.json] [--since 2026-11-12]

Cross-run: each hold run with --mass is paired with the latest hold run without a mass on the same joint; the
PWM difference per angle against M g arm gives duty per N m, so tau_stall = 1 / slope (N m at 100 % PWM, at that V).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_analysis as ba  # noqa: E402


def torque_cal(holds):
    out = []
    for h in holds:
        if not h["mass"]:
            continue
        ref = [x for x in holds if not x["mass"] and x["joint"] == h["joint"] and x["stem"] < h["stem"]]
        if not ref:
            out.append({"mass_run": h["stem"], "error": "no hold run without a mass on %s before it" % h["joint"]})
            continue
        r = ref[-1]
        a0 = {x["angle"]: x for x in r["table"]}
        dd, tq = [], []
        for x in h["table"]:
            y = a0.get(x["angle"])
            if y is None:
                continue
            dd.append(x["pwm_mean"] - y["pwm_mean"])
            tq.append(h["mass"] * 9.81 * x["arm_m"])
        if len(dd) < 3:
            out.append({"mass_run": h["stem"], "error": "fewer than 3 common angles"})
            continue
        dd, tq = np.array(dd), np.array(tq)
        k = float(np.dot(tq, dd) / np.dot(tq, tq))  # duty per N m (through the origin)
        out.append({"mass_run": h["stem"], "ref_run": r["stem"], "mass": h["mass"], "label": h["mass_label"],
                    "duty_per_Nm": round(k, 4), "tau_stall_Nm": round(abs(1.0 / k), 3) if k else None,
                    "fit_r2": round(1 - np.sum((dd - k * tq) ** 2) / max(np.sum((dd - dd.mean()) ** 2), 1e-12), 3),
                    "V": h["V_median"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=os.path.expanduser("~/hexapod_logs/bench"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--since", default="", help="only stems whose name sorts after this prefix (e.g. 2026-11-12)")
    args = ap.parse_args()
    stems = sorted(p[:-4] for p in glob.glob(os.path.join(args.dir, "*.npz")) if os.path.exists(p[:-4] + ".json"))
    stems = [s for s in stems if os.path.basename(s) >= args.since]
    res = []
    for s in stems:
        try:
            o = ba.analyse(s)
        except Exception as e:
            print("skip %s: %s" % (os.path.basename(s), e))
            continue
        o["stem"] = os.path.basename(s)
        res.append(o)
        print(ba.format_summary(o))
    cal = torque_cal([o for o in res if o["test"] == "hold"])
    if cal:
        print("== PWM -> torque calibration")
        for c in cal:
            print("  %s" % c)
    out = args.out or os.path.join(args.dir, "bench_summary.json")
    with open(out, "w") as f:
        json.dump({"runs": res, "torque_calibration": cal}, f, indent=1, default=str)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
