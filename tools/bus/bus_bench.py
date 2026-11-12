#!/usr/bin/env python3
"""Read-only bus bench for the teleop runner (Pi, Python 3.8). XL430-W250, Protocol 2.0. Torque stays OFF; the only
write is a SYNC_WRITE of each servo's own present position to Goal Position (RAM), the packet the runner sends per step.

    python3 tools/bus/bus_bench.py [--steps 500] [--sync-timeout 0.010] [--no-imu]

Reports per servo: model, firmware, Return Delay, drive/operating mode, position limits vs the contract clip box, PWM
limit, position PID + feed-forward, profile, voltage, temperature, hardware error. Then times N 50 Hz steps of
  1 x SYNC_READ Present Position (18)  +  1 x health read  +  IMU read  +  1 x SYNC_WRITE Goal Position
and prints p50 / p99 / max per part, the dropped-read rate, and the budget left for inference (6.4 ms) at 20 ms.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import dxl_bus as dx  # noqa: E402
from joint_map import Contract  # noqa: E402

INFER_MS = 6.4


def pct(x):
    x = np.asarray(x) * 1e3
    return "p50 %6.2f  p99 %6.2f  max %6.2f ms" % (np.percentile(x, 50), np.percentile(x, 99), x.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyDXL")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sync-timeout", type=float, default=0.010)
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--with-policy", action="store_true", help="run the ONNX policy each step (runner order: read, "
                    "IMU, infer, write, health every 2nd step)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--imu-thread", action="store_true", help="IMU read in a worker thread, overlapped with the "
                    "SYNC_READ (as the runner does); 'IMU read' is then the extra wait after the servo read")
    args = ap.parse_args()
    c = Contract()
    bus = dx.DxlBus(args.port, sync_timeout=args.sync_timeout)
    ids = [int(i) for i in c.ids]
    try:
        bus.torque_off_all()
        models = {i: bus.ping(i) for i in ids}
        missing = [c.names[k] for k, i in enumerate(ids) if models[i] is None]
        print("ping: %d/18 answer%s" % (18 - len(missing), (" - MISSING: " + ", ".join(missing)) if missing else ""))
        if len(missing) == 18:
            raise SystemExit("no servo answers: servo power off (battery switch)? wrong port/baud?")
        print("%-20s %3s %5s %3s %2s %2s %2s %11s %11s %4s %16s %9s %5s %5s %3s" % (
            "joint", "id", "model", "fw", "RD", "dm", "om", "min..max", "clip ticks", "pwm", "P/I/D ff2/ff1",
            "prof a/v", "V", "T", "err"))
        problems = []
        for k, i in enumerate(ids):
            r = lambda a, n: bus.read(i, a, n)  # noqa: E731
            fw, rd, dm, om = r(dx.FIRMWARE, 1), r(dx.RETURN_DELAY, 1), r(dx.DRIVE_MODE, 1), r(dx.OPERATING_MODE, 1)
            lo_p, hi_p, pwm = r(dx.MIN_POSITION, 4), r(dx.MAX_POSITION, 4), r(dx.PWM_LIMIT, 2)
            pid = (r(dx.POS_P, 2), r(dx.POS_I, 2), r(dx.POS_D, 2), r(dx.FF2, 2), r(dx.FF1, 2))
            prof = (r(dx.PROFILE_ACCEL, 4), r(dx.PROFILE_VELOCITY, 4))
            h, hw = bus.health(i), r(dx.HW_ERROR, 1)
            lo, hi = dx.rad_to_pos(c.lo[k]), dx.rad_to_pos(c.hi[k])
            print("%-20s %3d %5s %3s %2s %2s %2s %5s..%-4s %5d..%-4d %4s %4s/%s/%s %s/%s %4s/%-4s %5s %5s %3s" % (
                c.names[k], i, models[i], fw, rd, dm, om, lo_p, hi_p, lo, hi, pwm, pid[0], pid[1], pid[2], pid[3],
                pid[4], prof[0], prof[1], h and "%.1f" % h[0], h and "%.0f" % h[1], hw))
            if models[i] != dx.MODEL_XL430:
                problems.append("%s: model %s (expected %d)" % (c.names[k], models[i], dx.MODEL_XL430))
            if rd != 0:
                problems.append("%s: Return Delay %s (need 0)" % (c.names[k], rd))
            if om != dx.OP_POSITION:
                problems.append("%s: Operating Mode %s (need 3)" % (c.names[k], om))
            if lo_p is None or hi_p is None or lo_p > lo or hi_p < hi:
                problems.append("%s: position limits %s..%s do not cover the clip box %d..%d"
                                % (c.names[k], lo_p, hi_p, lo, hi))
            if hw:
                problems.append("%s: hardware error 0x%02x (%s)" % (c.names[k], hw, dx.errors_str(hw)))
        print("\n".join(["PROBLEM: " + p for p in problems]) or "static checks: all OK")

        imu = None
        if not args.no_imu:
            from imu_block import MPU6050
            imu = MPU6050()
        pol = None
        if args.with_policy:
            from s1h_policy import ObsBuilder, S1HPolicy
            pol, ob = S1HPolicy(c, threads=args.threads), ObsBuilder(c)
            a_prev = np.zeros(18)
        reads, health, write, imut, total, infer = [], [], [], [], [], []
        drops = np.zeros(18, dtype=int)
        pool = None
        if args.imu_thread and imu is not None:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=1)
        nxt = time.monotonic()
        for s in range(args.steps):
            t0 = time.perf_counter()
            fut = pool.submit(imu.read) if pool is not None else None
            q = bus.read_positions(ids)
            t1 = time.perf_counter()
            if fut is not None:
                fut.result(timeout=0.05)
            elif imu is not None:
                imu.read()
            t2 = time.perf_counter()
            ok = np.array([v is not None for v in q])
            drops += ~ok
            qa = np.array([v if v is not None else 0.0 for v in q])
            if pol is not None:
                o_p = ob.frame(qa, np.zeros(18), ok.astype(float), c.q_default, a_prev,
                               np.array([0, 0, 9.81, 0, 0, 0, 0, 0, -1.0]))
                a_prev = pol.act(o_p, np.zeros(3))
            t3 = time.perf_counter()
            if ok.all():
                bus.write_goals(ids, qa)  # torque is off: only the RAM goal register changes
            t4 = time.perf_counter()
            if s % 2 == 0:
                bus.health(ids[(s // 2) % 18])
            t5 = time.perf_counter()
            reads.append(t1 - t0)
            imut.append(t2 - t1)
            infer.append(t3 - t2)
            write.append(t4 - t3)
            health.append(t5 - t4)
            total.append(t5 - t0)
            nxt += c.dt
            w = nxt - time.monotonic()
            if w > 0:
                time.sleep(w)
        n = 18 * args.steps
        print("\n%d steps at 50 Hz, sync-read timeout %.1f ms:" % (args.steps, args.sync_timeout * 1e3))
        print("  SYNC_READ 18 positions %s" % pct(reads))
        print("  IMU read               %s" % pct(imut))
        if pol is not None:
            print("  policy inference       %s" % pct(infer))
        print("  SYNC_WRITE goals       %s" % pct(write))
        print("  health (every 2nd)     %s" % pct(health))
        print("  step total             %s" % pct(total))
        print("  dropped reads: %d / %d (%.2f %%), per servo: %s" % (drops.sum(), n, 100.0 * drops.sum() / n,
                                                                   drops.tolist()))
        p99 = np.percentile(total, 99) * 1e3
        extra = 0.0 if pol is not None else INFER_MS
        tot = p99 + extra
        verdict = "FITS with 15 % margin" if tot < 17.0 else "FITS (no 15 % margin)" if tot < 20.0 else "DOES NOT FIT"
        print("  budget: step p99 %.1f%s = %.1f ms of 20 ms -> %s" % (
            p99, "" if pol is not None else " + inference %.1f (estimate)" % INFER_MS, tot, verdict))
    finally:
        bus.torque_off_all()
        bus.close()


if __name__ == "__main__":
    main()
