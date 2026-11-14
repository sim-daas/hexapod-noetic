#!/usr/bin/env python3
"""XL430-W250 bench + ground tests for the hexasim motor model (Pi, Python 3.8). No policy, no ROS, no xs_sdk.

Every test logs raw samples (one servo read at ~0.7-1 kHz, or all 18 at 50 Hz) to ~/hexapod_logs/bench/<stamp>_<test>.npz
plus a .json with the test definition, then prints a summary (tools/bench/bench_analysis.py). The raw logs are what
hexasim replays to fit actuators/xl430.py; the summaries are a first look.

    python3 tools/bench/servo_bench.py status                 # battery voltage, temps, latched alarms, modes (no torque)
    python3 tools/bench/servo_bench.py hung                   # ROBOT HUNG UP, legs free: speed, step, sine, hold, drops
    python3 tools/bench/servo_bench.py hold --mass 0.52 --mass-label "500 ml bottle"   # hung, weight on the RF toe
    python3 tools/bench/servo_bench.py stall --joint LB_femur  # robot on its belly, body held down, foot pushes the floor
    python3 tools/bench/servo_bench.py ground                 # robot on its belly on the floor: stand, loads, tripods, sit

Single tests (hung): speed | step | sine | hold | drops, each with --joint. --sim runs everything on a simulated XL430
(offline check of the code paths; tools/bench/test_bench_offline.py). --yes skips the confirmation prompts.

Setup of the hung tests: the robot hangs (strap/string under the body or a stand) with all six feet clear of anything
for >= 25 cm, roughly level. The test leg sweeps: coxa +-0.45 rad, femur/tibia +-0.4 rad around the stand pose.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_lib as bl  # noqa: E402
from bench_lib import Abort, Recorder, save_run  # noqa: E402
import bench_analysis as ba  # noqa: E402

import dxl_bus as dx  # noqa: E402
from leg_ik import LegKin  # noqa: E402
from poses import STEP_ORDER, Motion, Poses  # noqa: E402

DT = 0.02


# ====================================================================================================== helpers
def meta_base(b, test, **kw):
    m = {"test": test, "kp": bl.KP, "pwm_full": bl.PWM_FULL, "ids": b.ids, "names": b.c.names, "sim": b.s.sim}
    m.update(kw)
    return m


def finish(b, name, rec, meta, summarise=True):
    if len(rec) == 0:
        b.say("%s: no samples recorded" % name)
        return None
    stem = save_run(name, rec, meta, b.logdir)
    b.saved.append(stem)
    b.say("saved %s.{npz,json} (%d samples)" % (stem, len(rec)))
    if summarise:
        try:
            out = ba.analyse(stem)
            b.say(ba.format_summary(out))
        except Exception as e:  # analysis must never lose a run
            b.say("summary failed (%s); the raw log is saved" % e)
    return stem


def sample(b, i, rec=None, misses=None, **fields):
    s = b.s.block(i)
    if s is None:
        if misses is not None:
            misses[0] += 1
            if misses[0] > 50:
                raise Abort("id %d stopped answering" % i)
        return None
    if misses is not None:
        misses[0] = 0
    if rec is not None:
        rec.add(t=s["t"], q=s["q"], vel=s["vel"], pwm=s["pwm"], load=s["load"], V=s["V"], T=s["T"], **fields)
    return s


def hang_pose(b):
    return b.c.q_default.copy()


def prepare_hung(b, cap=bl.CAP_SCRIPT):
    if not b.ask("Robot HUNG UP, all feet clear (>= 25 cm), nobody's hands near the legs? [y/N] "):
        raise Abort("not confirmed")
    q = b.torque_up(cap)
    q_h = hang_pose(b)
    b.ramp(q, q_h, rate=0.4)
    return q_h


def survey(b, what=""):
    v, vmin, tmax, alarms = b.voltage_survey()
    b.say("%sbattery %.2f V (median of 18 servos, min %.1f), hottest servo %.0f C%s" % (
        what, v, vmin, tmax, ("; LATCHED ALARMS: %s" % {i: dx.errors_str(h) for i, h in alarms.items()}) if alarms
        else ""))
    if v < bl.V_LOW_WARN:
        b.say("WARNING: battery below %.1f V - charge it; motor numbers scale with voltage" % bl.V_LOW_WARN)
    return v, tmax, alarms


# ====================================================================================================== status
def t_status(b, args):
    b.restore_marked()
    rows = []
    for k, i in enumerate(b.ids):
        h = b.s.health(i)
        rows.append((b.c.short(k), i, b.s.read(i, dx.OPERATING_MODE, 1), b.s.read(i, dx.TORQUE_ENABLE, 1),
                     h and h[0], h and h[1], b.s.hw_error(i)))
    b.say("%-10s %3s %4s %3s %6s %4s %s" % ("joint", "id", "mode", "trq", "V", "T", "hw error"))
    for r in rows:
        b.say("%-10s %3d %4s %3s %6s %4s %s" % (r[0], r[1], r[2], r[3], r[4], r[5],
                                               "-" if not r[6] else "0x%02x %s" % (r[6], dx.errors_str(r[6]))))
    v, _, alarms = survey(b)
    if alarms and (args.clear or b.ask("Reboot the servos with latched alarms (clears them)? [y/N] ")):
        b.clear_alarms()
    return v


# ====================================================================================================== speed (PWM mode)
def t_speed(b, args, k=None):
    """Free-running speed vs duty in PWM mode (Operating Mode 16), gravity-free coxa. Per duty: drive across the
    window, plug-brake at the edge, rest; alternate direction. Gives w_nl(V), the Coulomb duty and the motor time
    constant (J w_nl / tau_s)."""
    k = b.joint(args.joint or "RF_coxa") if k is None else k
    i = b.ids[k]
    duties = [float(x) for x in args.duties.split(",")]
    win, guard = args.window, args.window + 0.15
    c = hang_pose(b)[k]
    rec, segs, misses = Recorder(), [], [0]
    b.say("speed: %s, PWM mode, duties %s, window +-%.2f rad around %.2f" % (b.label(k), duties, win, c))
    b.to_pwm_mode(i)
    b.s.torque([i], True)
    seg = 0
    try:
        for d in duties:
            for p in range(args.passes):
                s = sample(b, i, misses=misses)
                while s is None:
                    s = sample(b, i, misses=misses)
                sgn = 1.0 if s["q"] <= c else -1.0
                edge, q0, t0 = c + sgn * 0.8 * win, s["q"], s["t"]
                b.s.duty(i, sgn * d)
                stuck = False
                while True:  # drive
                    s = sample(b, i, rec, misses, seg=seg, phase=1, cmd=sgn * d)
                    if s is None:
                        continue
                    b.check(s)
                    if abs(s["q"] - c) > guard:
                        raise Abort("left the +-%.2f rad guard (q %.2f)" % (guard, s["q"]))
                    if (s["q"] - edge) * sgn >= 0:
                        break
                    if s["t"] - t0 > 2.5 and abs(s["q"] - q0) < 0.03:
                        stuck = True
                        break
                    if s["t"] - t0 > 8.0:
                        break
                db = -sgn * max(d, 0.3)
                b.s.duty(i, db)
                tb = b.s.now()
                while b.s.now() - tb < 0.6:  # plug-brake until the joint stops
                    s = sample(b, i, rec, misses, seg=seg, phase=2, cmd=db)
                    if s is None:
                        continue
                    b.check(s)
                    if abs(s["q"] - c) > guard:
                        raise Abort("left the +-%.2f rad guard while braking (q %.2f)" % (guard, s["q"]))
                    if s["vel"] * sgn <= 0.05:
                        break
                b.s.duty(i, 0.0)
                tr = b.s.now()
                while b.s.now() - tr < 0.3:
                    sample(b, i, rec, misses, seg=seg, phase=3, cmd=0.0)
                segs.append({"seg": seg, "duty": d, "sign": sgn, "stuck": stuck})
                seg += 1
                if stuck:
                    b.say("  duty %.2f: no motion (static friction above this duty)" % d)
                    break
    finally:
        b.s.duty(i, 0.0)
        b.to_position_mode(i)
        q = b.read_q()
        b.s.goals([i], [q[k]])
        b.s.torque([i], True)
        b.ramp(q, hang_pose(b), rate=0.4)
        finish(b, "speed_%s" % b.c.short(k).replace(" ", "_"), rec,
               meta_base(b, "speed", joint=k, id=i, center=c, window=win, segs=segs, duties=duties))


# ====================================================================================================== step
def t_step(b, args, k=None):
    """Position steps at the runner's gains (P 1061, D 0, no profile) for each Goal PWM cap."""
    k = b.joint(args.joint or "RF_coxa") if k is None else k
    i = b.ids[k]
    c = hang_pose(b)[k]
    amps = [float(x) for x in args.amps.split(",")]
    caps = [int(x) for x in args.caps.split(",")]
    rec, segs, misses = Recorder(), [], [0]
    seg = 0
    b.say("step: %s, amplitudes %s rad, caps %s, hold %.1f s" % (b.label(k), amps, caps, args.hold))
    try:
        for cap in caps:
            b.s.cap([i], cap)
            for a in amps:
                for sgn in (1.0, -1.0):
                    for g in (c + sgn * a, c):
                        g = float(np.clip(g, b.c.lo[k], b.c.hi[k]))
                        s = sample(b, i, misses=misses)
                        b.s.goals([i], [g])
                        tw = b.s.now()
                        segs.append({"seg": seg, "cap": cap, "goal": g, "from": s["q"] if s else float("nan"),
                                     "t_write": tw, "amp": a, "sign": sgn if g != c else -sgn})
                        while b.s.now() - tw < args.hold:
                            s = sample(b, i, rec, misses, seg=seg, cmd=g, cap=cap)
                            if s is not None:
                                b.check(s)
                        seg += 1
    finally:
        b.s.cap([i], bl.CAP_SCRIPT)
        b.s.goals([i], [c])
        finish(b, "step_%s" % b.c.short(k).replace(" ", "_"), rec,
               meta_base(b, "step", joint=k, id=i, center=c, segs=segs, sat_err=bl.SAT_ERR))


# ====================================================================================================== sine
def t_sine(b, args, k=None):
    """Sine targets written at 50 Hz (zero-order hold, as the runner/sim), amplitude A, per frequency."""
    k = b.joint(args.joint or "RF_coxa") if k is None else k
    i = b.ids[k]
    c = hang_pose(b)[k]
    freqs = [float(x) for x in args.freqs.split(",")]
    A = args.sine_amp
    b.s.cap([i], args.sine_cap)
    rec, segs, misses = Recorder(), [], [0]
    b.say("sine: %s, A %.3f rad, f %s Hz, cap %d" % (b.label(k), A, freqs, args.sine_cap))
    try:
        for seg, f in enumerate(freqs):
            T = max(3.0, 8.0 / f)
            t0 = b.s.now()
            nxt, n, g = t0, 0, c
            while b.s.now() - t0 < T:
                if b.s.now() >= nxt:
                    g = c + A * math.cos(2.0 * math.pi * f * n * DT)
                    b.s.goals([i], [g])
                    n += 1
                    nxt += DT
                s = sample(b, i, rec, misses, seg=seg, cmd=g, f=f)
                if s is not None:
                    b.check(s)
            segs.append({"seg": seg, "f": f, "T": T})
            b.s.goals([i], [c])
            tr = b.s.now()
            while b.s.now() - tr < 0.5:
                sample(b, i, misses=misses)
    finally:
        b.s.cap([i], bl.CAP_SCRIPT)
        b.s.goals([i], [c])
        finish(b, "sine_%s" % b.c.short(k).replace(" ", "_"), rec,
               meta_base(b, "sine", joint=k, id=i, center=c, A=A, cap=args.sine_cap, segs=segs, dt=DT))


# ====================================================================================================== hold (static load)
def t_hold(b, args, k=None):
    """Static holds of a femur at n angles, each approached from below and from above. Present PWM vs angle gives the
    leg's gravity moment, friction (half the up/down difference) and the P-law check (PWM vs position error).
    With --mass M (kg) hung from the toe, the PWM difference to a run without it calibrates PWM -> torque, so
    tau_stall(V) = M g arm / delta_duty (arm from FK, the toe's horizontal distance from the femur axis)."""
    k = b.joint(args.joint or "RF_femur") if k is None else k
    if not 6 <= k < 12:
        raise SystemExit("hold: pick a femur (the static-gravity geometry assumes it)")
    i = b.ids[k]
    leg = k - 6
    kin = LegKin(b.c.kin)
    q_t = hang_pose(b)[12 + leg]
    center, span, n = args.hold_center, args.hold_span, args.hold_n
    angles = np.linspace(center - span, center + span, n)
    r, _z = kin.plane_fk(angles, np.full(n, q_t))
    arm = r - kin.r0
    if args.mass:
        tau = args.mass * 9.81 * np.abs(arm)
        if tau.max() > 0.75:
            raise SystemExit("hold: %.2f kg at the toe gives up to %.2f N m (> 0.75 = 54 %% of stall); use a lighter "
                             "weight or a smaller --hold-span" % (args.mass, tau.max()))
        b.say("hold: weight %.3f kg (%s) at the toe: %.2f..%.2f N m" % (args.mass, args.mass_label, tau.min(),
                                                                       tau.max()))
    rec, segs, misses = Recorder(), [], [0]
    q_h = hang_pose(b)
    q_h[k] = center
    q = b.read_q()
    b.ramp(q, q_h, rate=0.4)
    if args.mass and not b.ask("Hang the weight from the %s toe now, let it settle, then confirm [y/N] "
                               % b.c.short(leg).split()[0]):
        raise Abort("not confirmed")
    b.say("hold: %s, %d angles %.2f..%.2f rad, tibia %.2f, from below and above" % (
        b.label(k), n, angles[0], angles[-1], q_t))
    seg = 0
    try:
        for a in angles:
            for side in (-1.0, 1.0):
                pre = float(a + side * 0.12)
                b.s.goals([i], [pre])
                tw = b.s.now()
                while b.s.now() - tw < 0.6:
                    sample(b, i, rec, misses, seg=-1, cmd=pre)
                b.s.goals([i], [float(a)])
                tw = b.s.now()
                while b.s.now() - tw < args.hold_s:
                    s = sample(b, i, rec, misses, seg=seg, cmd=float(a))
                    if s is not None:
                        b.check(s)
                segs.append({"seg": seg, "angle": float(a), "from": side, "arm": float(kin.plane_fk(a, q_t)[0] - kin.r0)})
                seg += 1
    finally:
        # save first: after a latched alarm every ramp step re-raises Abort, which used to lose the whole run
        name = "hold_%s%s" % (b.c.short(k).replace(" ", "_"), "_mass" if args.mass else "")
        finish(b, name, rec, meta_base(b, "hold", joint=k, id=i, tibia=q_t, segs=segs, mass=args.mass or 0.0,
                                       mass_label=args.mass_label, hold_s=args.hold_s, sat_err=bl.SAT_ERR))
        try:
            b.ramp(b.read_q(), hang_pose(b), rate=0.4)
        except Abort as e:
            b.say("hold: return ramp skipped (%s)" % e)
        if args.mass:
            b.ask("Remove the weight, then confirm [y/N] ")


# ====================================================================================================== drops
def t_drops(b, args):
    """All 18 servos zig-zag around the hang pose (|delta q_cmd| = amp per 50 Hz step, sign flips with p 0.7, the
    policy's chatter) while the runner's read (SYNC_READ + retry + adaptive order) runs: dropped reads vs step size."""
    amps = [float(x) for x in args.drop_amps.split(",")]
    caps = [int(x) for x in args.caps.split(",")]
    q_h = hang_pose(b)
    rng = np.random.RandomState(1)
    rec, segs = Recorder(), []
    diag_src = getattr(b.s, "bus", b.s)
    seg = 0
    b.say("drops: all 18 servos, |dq_cmd| per step %s rad, caps %s, %.0f s each" % (amps, caps, args.drop_s))
    b.ramp(b.read_q(), q_h, rate=0.4)
    try:
        for cap in caps:
            b.s.cap(b.ids, cap)
            for a in amps:
                sgn = np.where(rng.rand(18) < 0.5, 1.0, -1.0)
                nxt = b.s.now()
                for st in range(int(args.drop_s / DT)):
                    flip = rng.rand(18) < 0.7
                    sgn = np.where(flip, -sgn, sgn)
                    q_cmd = b.c.clip(q_h + 0.5 * a * sgn)
                    b.s.goals(b.ids, q_cmd)
                    q = b.s.read_positions(b.ids)
                    t = b.s.now()
                    valid = np.array([v is not None for v in q])
                    V = float("nan")
                    if st % 2 == 0:
                        h = b.s.health(b.ids[(st // 2) % 18])
                        if h:
                            V = h[0]
                            if V < bl.V_ABORT:
                                raise Abort("voltage %.1f V" % V)
                    b.check()
                    rd = getattr(diag_src, "last_read_diag", (0, 0, 0, 0, 0))
                    rec.add(t=t, seg=seg, q_cmd=q_cmd, q=np.array([v if v is not None else np.nan for v in q]),
                            valid=valid, V=V, rd=np.array(rd, dtype=float))
                    nxt += DT
                    w = nxt - b.s.now()
                    if w > 0:
                        b.s.sleep(w)
                segs.append({"seg": seg, "amp": a, "cap": cap})
                seg += 1
                b.s.goals(b.ids, q_h)
                b.hold_all(q_h, 1.0)
    finally:
        b.s.cap(b.ids, bl.CAP_SCRIPT)
        finish(b, "drops", rec, meta_base(b, "drops", segs=segs, dt=DT))


# ====================================================================================================== stall / overload
def t_stall(b, args):
    """Overload protection: the femur pushes its foot into the floor (robot on its belly, body held down) with the P loop asking req x 100 %
    PWM under a Goal PWM cap; time until the overload alarm (or timeout). Latched alarms are cleared by reboot."""
    k = b.joint(args.joint or "LB_femur")
    if not 6 <= k < 12:
        raise SystemExit("stall: pick a femur")
    i, kt = b.ids[k], k + 6
    it = b.ids[kt]
    d = -1.0 if args.push == "up" else 1.0  # femur + moves the toe down
    lo, hi = b.c.lo[k], b.c.hi[k]
    trials = []
    for tok in args.trials.split(","):
        cap, req = tok.split(":")
        trials.append((float(cap), float(req)))
    if args.push == "down":
        msg = ("Robot on its BELLY on the floor. The %s foot will press into the floor: hold the body down with both "
               "hands on the far side of the body (away from the legs) or put >= 3 kg on it. Ready? [y/N] ")
    else:
        msg = "Robot on its BELLY, the %s foot trapped under a heavy object (toe pushes up). Ready? [y/N] "
    if not b.ask(msg % b.c.short(k).split()[0]):
        raise Abort("not confirmed")
    q0 = b.torque_up(bl.CAP_SCRIPT)
    b.s.cap([it], bl.PWM_FULL)  # the tibia must not be the joint that gives
    rec, segs, misses = Recorder(), [], [0]
    # ---- contact search: 0.1 rad/s in the push direction at 30 % until the error passes 0.05 rad
    b.s.cap([i], int(0.30 * bl.PWM_FULL))
    g, q_c = q0[k], None
    nxt = b.s.now()
    while abs(g - q0[k]) < args.search:
        g = float(np.clip(g + d * 0.1 * DT, lo, hi))
        if g in (lo, hi):
            break
        b.s.goals([i], [g])
        nxt += DT
        while b.s.now() < nxt:
            s = sample(b, i, rec, misses, seg=-1, cmd=g, cap=0.30, tib_pwm=float("nan"))
            if s is not None:
                b.check(s)
                if (g - s["q"]) * d > 0.05:
                    q_c = s["q"]
                    break
        if q_c is not None:
            break
    if q_c is None:
        b.s.goals([i], [q0[k]])
        raise Abort("no obstacle within %.2f rad of the start: is the foot trapped?" % args.search)
    b.say("stall: contact at %s = %.3f rad (moved %.3f rad)" % (b.label(k), q_c, q_c - q0[k]))
    rel = q_c - d * 0.03
    b.s.goals([i], [rel])
    T_base = None
    seg = 0
    try:
        for cap, req in trials:
            # cool down to within 3 C of the first trial's start temperature
            tw = b.s.now()
            while True:
                s = sample(b, i, misses=misses)
                if s is None:
                    continue
                T_base = s["T"] if T_base is None else T_base
                if s["T"] <= T_base + 3.0 or b.s.now() - tw > args.cool_max:
                    break
                b.s.sleep(1.0)
            b.s.cap([i], int(round(cap * bl.PWM_FULL)))
            goal = float(np.clip(q_c + d * req * bl.SAT_ERR, lo, hi))
            b.s.goals([i], [goal])
            b.s.take_alerts()
            t0 = b.s.now()
            outcome, t_trip, hw, n, tib_hi = "timeout", float("nan"), 0, 0, 0.0
            b.say("  trial cap %3.0f %%, P asks %3.0f %%: pushing ..." % (cap * 100, req * 100))
            while True:
                s = sample(b, i, rec, misses, seg=seg, cmd=goal, cap=cap, tib_pwm=float("nan"))
                n += 1
                al = b.s.take_alerts()
                if (s is not None and s["alert"]) or i in al:
                    outcome, t_trip = "trip", (s["t"] if s else b.s.now()) - t0
                    hw = b.s.hw_error(i) or 0
                    break
                if al:
                    raise Abort("alert on another servo: %s" % sorted(al))
                if s is None:
                    continue
                if s["T"] >= bl.TEMP_ABORT:
                    outcome = "temp"
                    break
                if 0 < s["V"] < bl.V_ABORT:
                    raise Abort("voltage %.1f V" % s["V"])
                if n % 10 == 0:
                    st = b.s.block(it)
                    if st is not None:
                        rec.add(t=st["t"], q=s["q"], vel=s["vel"], pwm=s["pwm"], load=s["load"], V=s["V"], T=s["T"],
                                seg=seg, cmd=goal, cap=cap, tib_pwm=st["pwm"])
                        tib_hi = tib_hi + 10 * 0.0013 if abs(st["pwm"]) > 0.95 else 0.0
                        if tib_hi > 1.0:
                            outcome = "tibia saturated"
                            break
                if s["t"] - t0 > args.trial_max:
                    break
            b.s.goals([i], [rel])
            segs.append({"seg": seg, "cap": cap, "req": req, "goal": goal, "outcome": outcome, "t_trip": t_trip,
                         "hw": hw, "q_contact": q_c})
            b.say("  -> %s%s" % (outcome, " after %.2f s (hw 0x%02x %s)" % (t_trip, hw, dx.errors_str(hw))
                                 if outcome == "trip" else ""))
            if outcome == "trip":
                b.clear_alarms(ids=[i])
                q = b.read_q()
                b.s.goals([i], [q[k]])
                b.s.torque([i], True)
                b.ramp(q, np.where(np.arange(18) == k, rel, q), duration=1.0)
            tr = b.s.now()
            while b.s.now() - tr < args.rest:
                sample(b, i, rec, misses, seg=-2, cmd=rel, cap=bl.CAP_SCRIPT / bl.PWM_FULL, tib_pwm=float("nan"))
                b.s.sleep(0.2)
            seg += 1
    finally:
        b.s.cap([i, it], bl.CAP_SCRIPT)
        b.s.goals([i], [q0[k]])
        # save before hold_all: its check() re-raises a latched alarm and would lose the run
        finish(b, "stall_%s" % b.c.short(k).replace(" ", "_"), rec,
               meta_base(b, "stall", joint=k, id=i, tibia_id=it, push=args.push, segs=segs, sat_err=bl.SAT_ERR))
        try:
            b.hold_all(b.read_q(), 0.2)
        except Abort as e:
            b.say("stall: final hold skipped (%s)" % e)


# ====================================================================================================== ground
class Stall(Exception):
    pass


class StallMonitor:
    """|Present PWM| >= frac x Goal PWM cap for >= hold_s on any joint -> Stall (well before the ~4 s overload trip)."""

    def __init__(self, cap_units, frac=0.92, hold_s=0.5):
        self.cap, self.frac, self.hold_s = cap_units / float(bl.PWM_FULL), frac, hold_s
        self.since = np.full(18, np.nan)
        self.peak = np.zeros(18)

    def __call__(self, t, q_cmd, q, pwm):
        a = np.abs(np.nan_to_num(pwm))
        self.peak = np.maximum(self.peak, a)
        hi = a >= self.frac * self.cap
        self.since = np.where(hi, np.where(np.isnan(self.since), t, self.since), np.nan)
        over = hi & (t - np.nan_to_num(self.since, nan=t) >= self.hold_s)
        if over.any():
            raise Stall(int(np.argmax(over)))


def leg_lift_q(kin, q, leg, lift):
    q = np.array(q, float)
    r, z = kin.plane_fk(q[6 + leg], q[12 + leg])
    qf, qt = kin.plane_ik(r, z + lift)
    q[6 + leg], q[12 + leg] = float(qf), float(qt)
    return q


def leg_step_segments(kin, q_from, q_to, leg, lift=0.03, T=1.6):
    """Leg ``leg`` lifts by ``lift``, moves (joint space) to q_to's lifted pose, lowers; the others hold q_from/q_to.
    Returns three (q_start, q_end, duration) ramps."""
    idx = [leg, 6 + leg, 12 + leg]
    a = np.array(q_from, float)
    up_a = leg_lift_q(kin, a, leg, lift)
    up_b = up_a.copy()
    up_b[idx] = leg_lift_q(kin, q_to, leg, lift)[idx]
    down = up_b.copy()
    down[idx] = np.asarray(q_to, float)[idx]
    return [(a, up_a, 0.3 * T), (up_a, up_b, 0.5 * T), (up_b, down, 0.3 * T)]


def t_ground(b, args):
    """Robot on its belly on a flat floor. Stand up (the runner's sequence, 70 % cap), then with the stall monitor on:
    hold STAND; step the legs one at a time into the stance the S1H4 policy commanded on 2026-11-11 and hold it at
    Goal PWM 619 and 885; step back; lift each tripod 2 cm; sit down. Logs pwm/load/vel/q of all 18 at 50 Hz."""
    import json
    from hexapod_teleop import SLEEP_TOL
    poses, kin = Poses(b.c), LegKin(b.c.kin)
    st = json.load(open(os.path.join(HERE, "policy_stance_2026-11-11.json")))
    stance = b.c.clip(np.array([st["q_cmd"][n] for n in b.c.names]))
    q = b.read_q()
    far = np.abs(q - poses.SLEEP) > SLEEP_TOL
    if far.any():
        raise Abort("not lying in the SLEEP pose: %s" % [b.c.short(k) for k in np.where(far)[0]])
    if not b.ask("Robot on its BELLY on a flat floor, 40 cm clear around it, you ready to catch it? [y/N] "):
        raise Abort("not confirmed")
    imu = None
    if not b.s.sim and not args.no_imu:
        from imu_block import ImuBlock, MPU6050
        imu, blk = MPU6050(), ImuBlock(os.path.join(bl.REPO, "config", "imu_calib.json"))
    rec, phases = Recorder(), []
    tilt_max = [0.0]

    def mon_factory(cap):
        sm = StallMonitor(cap)

        def mon(t, q_cmd, q, pwm):
            if imu is not None:
                a, _w = blk.body(*imu.read())
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, a[2] / max(np.linalg.norm(a), 1e-6)))))
                tilt_max[0] = max(tilt_max[0], tilt)
                if tilt > args.tilt_max:
                    raise Abort("tilt %.0f deg > %.0f" % (tilt, args.tilt_max))
            sm(t, q_cmd, q, pwm)
        return mon

    current = {}

    def play(segments, phase, cap, name):
        mon = mon_factory(cap)
        m = Motion(segments, DT)
        current.update(m=m, name=name)
        nxt = b.s.now()
        t0 = nxt
        while True:
            qc = m.next()
            if qc is None:
                break
            b.step_all(qc, rec, phase, mon)
            nxt += DT
            w = nxt - b.s.now()
            if w > 0:
                b.s.sleep(w)
        phases.append({"phase": phase, "name": name, "cap": cap, "t0": t0, "t1": b.s.now()})

    def hold(qh, secs, phase, cap, name):
        b.s.cap(b.ids, cap)
        t0 = b.s.now()
        outcome = "ok"
        try:
            b.hold_all(qh, secs, rec, phase, monitor=mon_factory(cap))
        except Stall as e:
            outcome = "stall %s" % b.label(e.args[0])
            b.say("  %s: STALL on %s (|PWM| >= 92 %% of the cap for 0.5 s) - phase stopped" % (name, b.label(e.args[0])))
            b.s.cap(b.ids, bl.CAP_SCRIPT)
        phases.append({"phase": phase, "name": name, "cap": cap, "t0": t0, "t1": b.s.now(), "outcome": outcome})
        return outcome

    from poses import Segment, joint_ramp  # noqa: F401
    standing = False
    try:
        q = b.torque_up(bl.CAP_SCRIPT)
        b.say("ground: standing up (%.0f s)" % sum(s.duration for s in poses.standup(q)))
        play(poses.standup(q), 1, bl.CAP_SCRIPT, "standup")
        standing = True
        b.say("ground: STAND hold %.0f s" % args.stand_s)
        hold(poses.STAND, args.stand_s, 2, bl.CAP_SCRIPT, "stand")
        if not args.no_stance:
            b.say("ground: stepping into the 2026-11-11 policy stance, one leg at a time")
            cur = poses.STAND.copy()
            for leg in STEP_ORDER:
                segs = [joint_ramp("in_%d" % leg, a, c_, d_) for a, c_, d_ in leg_step_segments(kin, cur, stance, leg)]
                play(segs, 3, bl.CAP_SCRIPT, "into_stance")
                cur = segs[-1].q(1.0)
            for ph, cap in ((4, bl.CAP_SCRIPT), (5, bl.PWM_FULL)):
                b.say("ground: policy stance hold %.0f s at Goal PWM %d" % (args.stance_s, cap))
                hold(cur, args.stance_s, ph, cap, "stance_%d" % cap)
            b.s.cap(b.ids, bl.CAP_SCRIPT)
            b.say("ground: stepping back to STAND")
            for leg in STEP_ORDER:
                segs = [joint_ramp("out_%d" % leg, a, c_, d_) for a, c_, d_ in
                        leg_step_segments(kin, cur, poses.STAND, leg)]
                play(segs, 6, bl.CAP_SCRIPT, "out_of_stance")
                cur = segs[-1].q(1.0)
        if not args.no_tripod:
            for ph, legs, name in ((7, (0, 4, 2), "tripod_A_lifted (RF LM RB up)"), (8, (3, 1, 5), "tripod_B_lifted (LF RM LB up)")):
                up = poses.STAND.copy()
                for leg in legs:
                    up = leg_lift_q(kin, up, leg, 0.02)
                b.say("ground: %s for %.0f s" % (name, args.tripod_s))
                play([joint_ramp("lift", poses.STAND, up, 0.6)], ph, bl.CAP_SCRIPT, name + " up")
                hold(up, args.tripod_s, ph, bl.CAP_SCRIPT, name)
                play([joint_ramp("lower", up, poses.STAND, 0.6)], ph, bl.CAP_SCRIPT, name + " down")
        b.say("ground: sitting down")
        play(poses.sitdown_from_stand(), 9, bl.CAP_SCRIPT, "sitdown")
        standing = False
    except (Abort, Stall) as e:
        b.say("ground: ABORT (%s)" % (e if isinstance(e, Abort) else "stall %s" % b.label(e.args[0])))
        bad = b.tolerate_alarmed()
        if bad:
            b.say("ground: servos with a latched alarm (limp until rebooted): %s" % bad)
        if not standing and current.get("name") == "standup" and current["m"].i >= 1:
            b.say("ground: reversing the stand-up from where it stopped - be ready to catch")
            try:
                b.s.cap(b.ids, bl.CAP_SCRIPT)
                core_k = min(current["m"].i - 1, len(poses.core) - 1)
                play(poses.sitdown_from_core(core_k, b.read_q()), 9, bl.CAP_SCRIPT, "sitdown_after_abort")
            except Exception as e2:
                b.say("ground: reverse failed too (%s): torque off" % e2)
        if standing:
            b.say("ground: back to STAND, then sit down - be ready to catch")
            try:
                b.s.cap(b.ids, bl.CAP_SCRIPT)
                b.ramp(b.read_q(), poses.STAND, rate=0.4)
                play(poses.sitdown_from_stand(), 9, bl.CAP_SCRIPT, "sitdown_after_abort")
            except Exception as e2:
                b.say("ground: sit-down failed too (%s): torque off" % e2)
        raise
    finally:
        b.s.torque_off_all()
        finish(b, "ground", rec, meta_base(b, "ground", phases=phases, stance=stance, tilt_max=tilt_max[0],
                                           stance_source=st["source"]))


# ====================================================================================================== main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["status", "hung", "speed", "step", "sine", "hold", "drops", "stall", "ground"])
    ap.add_argument("--joint", help="e.g. RF_coxa, LB_femur or a servo id (default per test)")
    ap.add_argument("--leg", default="RF", help="hung: the leg whose coxa/femur/tibia are tested (default RF)")
    ap.add_argument("--sim", action="store_true", help="simulated XL430 bus (offline)")
    ap.add_argument("--yes", action="store_true", help="answer yes to every prompt")
    ap.add_argument("--clear", action="store_true", help="status: reboot servos with latched alarms without asking")
    ap.add_argument("--port", default="/dev/ttyDXL")
    ap.add_argument("--logdir", default=bl.LOGDIR)
    ap.add_argument("--only", help="hung: comma list of speed,step,sine,hold,drops (default all, in that order)")
    # speed
    ap.add_argument("--duties", default="0.05,0.08,0.12,0.16,0.2,0.3,0.4,0.5,0.6,0.8,1.0")
    ap.add_argument("--window", type=float, default=0.45)
    ap.add_argument("--passes", type=int, default=4)
    # step
    ap.add_argument("--amps", default="0.02,0.05,0.1,0.2,0.4")
    ap.add_argument("--caps", default="885,619")
    ap.add_argument("--hold", type=float, default=0.8, help="step: s per target")
    # sine
    ap.add_argument("--freqs", default="0.5,1.5,3,5,8,12.5,25")
    ap.add_argument("--sine-amp", type=float, default=0.05)
    ap.add_argument("--sine-cap", type=int, default=885)
    # hold
    ap.add_argument("--mass", type=float, default=0.0, help="hold: kg hung from the toe (0 = leg only)")
    ap.add_argument("--mass-label", default="")
    ap.add_argument("--hold-center", type=float, default=0.0, help="hold: femur centre angle (0 = leg sticking out)")
    ap.add_argument("--hold-span", type=float, default=0.4)
    ap.add_argument("--hold-n", type=int, default=7)
    ap.add_argument("--hold-s", type=float, default=1.6)
    # drops
    ap.add_argument("--drop-amps", default="0,0.02,0.05,0.1,0.15,0.2")
    ap.add_argument("--drop-s", type=float, default=8.0)
    # stall
    ap.add_argument("--push", choices=["up", "down"], default="down",
                    help="stall: down = toe into the floor (hold the body down); up only with room to the femur clip")
    ap.add_argument("--trials", default="0.3:0.45,0.5:0.75,0.7:1.0,1.0:0.6,1.0:1.3",
                    help="cap:request pairs (fractions of 100 %% PWM); request > cap saturates the cap")
    ap.add_argument("--trial-max", type=float, default=40.0)
    ap.add_argument("--rest", type=float, default=15.0)
    ap.add_argument("--cool-max", type=float, default=180.0)
    ap.add_argument("--search", type=float, default=0.5)
    # ground
    ap.add_argument("--stand-s", type=float, default=10.0)
    ap.add_argument("--stance-s", type=float, default=8.0)
    ap.add_argument("--tripod-s", type=float, default=3.0)
    ap.add_argument("--tilt-max", type=float, default=25.0)
    ap.add_argument("--no-stance", action="store_true")
    ap.add_argument("--no-tripod", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    return ap


def make_bench(args, servos=None):
    out = sys.stdout

    def say(m):
        out.write(m + "\n")
        out.flush()

    def ask(m):
        if args.yes:
            say(m + "y (--yes)")
            return True
        try:
            return input(m).strip().lower() in ("y", "yes")
        except EOFError:
            return False

    bus = None
    if servos is None:
        if args.sim:
            servos = bl.SimServos(bl.Contract())
        else:
            servos, bus = bl.open_servos(args.port)
    return bl.Bench(servos, say=say, ask=ask, logdir=args.logdir), bus


def run(args, servos=None):
    b, bus = make_bench(args, servos)
    gov_prev = None
    if not b.s.sim:
        from hexapod_teleop import set_governor
        gov_prev = set_governor("performance")
    try:
        if args.test == "status":
            t_status(b, args)
            return b
        b.restore_marked()
        v, tmax, alarms = survey(b, "start: ")
        if alarms:
            if not (args.clear or b.ask("Latched alarms: reboot those servos first? [y/N] ")):
                raise Abort("latched alarms present")
            b.clear_alarms()
        if args.test in ("speed", "step", "sine", "hold", "drops", "hung"):
            prepare_hung(b)
            legk = b.joint(args.leg + "_coxa")
            if args.test == "hung":
                todo = (args.only or "speed,step,sine,hold,drops").split(",")
                for name in todo:
                    survey(b, "before %s: " % name)
                    if name == "speed":
                        t_speed(b, args, legk)
                    elif name in ("step", "sine"):
                        for kk in (legk, legk + 6, legk + 12):
                            (t_step if name == "step" else t_sine)(b, args, kk)
                    elif name == "hold":
                        t_hold(b, args, legk + 6)
                    elif name == "drops":
                        t_drops(b, args)
            else:
                {"speed": t_speed, "step": t_step, "sine": t_sine, "hold": t_hold, "drops": lambda b_, a_: t_drops(b_, a_)}[
                    args.test](b, args)
            q = b.read_q()
            b.ramp(q, hang_pose(b), rate=0.4)
        elif args.test == "stall":
            t_stall(b, args)
        elif args.test == "ground":
            t_ground(b, args)
        survey(b, "end: ")
    except Abort as e:
        b.say("ABORTED: %s" % e)
        if not getattr(args, "_no_raise", False) and not b.s.sim:
            pass
        b.aborted = str(e)
    except KeyboardInterrupt:
        b.say("Ctrl-C: stopping")
        b.aborted = "Ctrl-C"
    finally:
        b.safe_stop()
        if gov_prev and gov_prev != "performance":
            from hexapod_teleop import set_governor
            set_governor(gov_prev)
        if bus is not None:
            bus.close()
        if b.saved:
            b.say("logs:\n  " + "\n  ".join(b.saved))
    return b


def main(argv=None):
    args = build_parser().parse_args(argv)
    b = run(args)
    sys.exit(1 if getattr(b, "aborted", None) else 0)


if __name__ == "__main__":
    main()
