#!/usr/bin/env python3
"""Offline tests of the servo bench on the simulated XL430 (no robot; workstation or Pi, Python 3.8+, NumPy).

    python3 tools/bench/test_bench_offline.py
"""
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_lib as bl  # noqa: E402
import bench_analysis as ba  # noqa: E402
import servo_bench as sb  # noqa: E402
import analyze_bench as ab  # noqa: E402
import dxl_bus as dx  # noqa: E402
from leg_ik import LegKin  # noqa: E402
from poses import STEP_ORDER, Poses  # noqa: E402

C = bl.Contract()
TMP = tempfile.mkdtemp(prefix="bench_test_")
bl.PWM_MARK = os.path.join(TMP, ".pwm_mode_ids.json")


def args_for(test, *extra):
    return sb.build_parser().parse_args([test, "--sim", "--yes", "--logdir", TMP] + list(extra))


def run(test, *extra, servos=None):
    a = args_for(test, *extra)
    return sb.run(a, servos or bl.SimServos(C))


def last(prefix):
    import glob
    p = sorted(glob.glob(os.path.join(TMP, "*_%s*.npz" % prefix)))
    assert p, "no log for %s" % prefix
    return p[-1][:-4]


# ------------------------------------------------------------------------------------------------------ tests
def test_status_clears_latched_alarm():
    s = bl.SimServos(C)
    s.alarm[11] = 0x20
    b = run("status", "--clear", servos=s)
    assert not getattr(b, "aborted", None)
    assert s.alarm.sum() == 0 and s.kp[11] == bl.KP


def test_latched_alarm_blocks_without_clear():
    s = bl.SimServos(C)
    s.alarm[11] = 0x20
    a = args_for("step", "--joint", "RF_coxa")
    a.yes = False
    b = sb.run(a, s)  # ask() -> input() -> EOF -> no
    assert getattr(b, "aborted", None) and "latched" in b.aborted


def test_pwm_mode_restored_after_crash():
    s = bl.SimServos(C)
    b = bl.Bench(s, C, say=lambda m: None)
    b.to_pwm_mode(10)
    assert os.path.exists(bl.PWM_MARK) and s.mode[0] == bl.OP_PWM
    b2 = bl.Bench(s, C, say=lambda m: None)  # "next start"
    b2.restore_marked()
    assert s.mode[0] == dx.OP_POSITION and not os.path.exists(bl.PWM_MARK)


def test_speed():
    b = run("speed", "--duties", "0.005,0.2,0.5,1.0", "--passes", "2")
    assert not getattr(b, "aborted", None), b.aborted
    o = ba.analyse(last("speed"))
    f = o["fit"]
    # sim: w_nl = 5.97 V/11.1, tau_c 0.02 of tau_s ~1.5 -> d_c ~0.013
    assert 5.0 < f["w_nl_rad_s"] < 7.5, f
    assert 0.0 <= f["coulomb_duty"] < 0.05, f
    assert 0.005 in o["no_motion_duties"]
    assert (bl.SimServos is not None) and all(m == dx.OP_POSITION for m in b.s.mode)


def test_step_and_sine():
    b = run("step", "--joint", "RF_femur", "--amps", "0.05,0.2", "--caps", "885,619", "--hold", "0.5")
    assert not getattr(b, "aborted", None), b.aborted
    o = ba.analyse(last("step"))
    r = o["by_cap_amp"]["cap885_amp0.20"]
    assert 5 < r["t90_ms"] < 200 and r["dead_ms"] < 10, r
    assert r["peak_pwm"] <= 1.0 and o["by_cap_amp"]["cap619_amp0.20"]["peak_pwm"] <= 0.70 + 1e-3
    b = run("sine", "--joint", "RF_coxa", "--freqs", "1.5,12.5,25")
    o = ba.analyse(last("sine"))
    g = [r["gain_vs_A"] for r in o["rows"]]
    assert g[0] > 0.8 and g[0] > g[1] > g[2], g


def test_hold_and_torque_calibration():
    s = bl.SimServos(C, gravity=0.08)
    b = run("hold", servos=s)
    assert not getattr(b, "aborted", None), b.aborted
    o = ba.analyse(last("hold_RF_femur"))
    assert abs(o["p_law_measured_over_predicted"] - 1.0) < 0.15, o
    # add a known toe weight to the sim: tau = -m g arm on the RF femur (+ = toe down)
    kin = LegKin(C.kin)
    m = 0.4

    class Weighted(bl.SimServos):
        def _step(self, h):
            r, _ = kin.plane_fk(self.q[6], self.q[12])
            self.g_extra = m * 9.81 * (r - kin.r0)
            w0 = self.w[6]
            super()._step(h)
            self.w[6] = self.w[6] + h * self.g_extra / self.J
            _ = w0
    s2 = Weighted(C, gravity=0.08)
    b = run("hold", "--mass", str(m), "--mass-label", "sim", servos=s2)
    assert not getattr(b, "aborted", None), b.aborted
    holds = [ba.analyse(last("hold_RF_femur_mass")), o]
    for h in holds:
        h["stem"] = os.path.basename(h["stem"])
    holds.sort(key=lambda h: h["stem"])
    cal = ab.torque_cal(holds)
    assert cal and "tau_stall_Nm" in cal[0], cal
    V = holds[-1]["V_median"]
    assert abs(cal[0]["tau_stall_Nm"] - 1.4 * V / 11.1) < 0.2, (cal, V)


def test_hold_refuses_heavy_mass():
    try:
        run("hold", "--mass", "2.0")
    except SystemExit as e:
        assert "lighter" in str(e)
        return
    raise AssertionError("2 kg at the toe was not refused")


def test_stuffed_status_parse():
    # load -1 (FF FF) then velocity 253 (FD 00 00 00): the servo stuffs an FD after FF FF FD
    import struct
    params = struct.pack("<hhii", -5, -1, 253, 2048)
    stuffed = bl.unstuff(b"") or dx._stuff(params)
    assert len(stuffed) == len(params) + 1
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, 7]) + struct.pack("<H", len(stuffed) + 4) + bytes([0x55, 0]) + stuffed
    pk = body + struct.pack("<H", dx.crc16(body))
    got = bl.parse_status_any(b"\x00" + pk + pk[:5])
    assert got[7] == (0, params), got
    assert dx.parse_status(pk, 12) == {}  # the runner's parser drops it (positions-only reads never stuff)


def test_stall_monitor_ignores_brief_peaks():
    m = sb.StallMonitor(619)
    pwm = np.zeros(18)
    for j in range(100):
        pwm[3] = 0.69 if j % 10 < 3 else 0.2  # 60 ms peaks
        m(j * 0.02, None, None, pwm)


def test_ground_abort_mid_standup_reverses():
    s = bl.SimServos(C)
    orig = s.read_all
    n = [0]

    def read_all(ids):
        n[0] += 1
        if n[0] == 300:  # ~6 s in: inside the push-up / re-step part of the core
            s.last_error[ids[0]] = dx.ALERT
            s.alarm[0] = 0x04
        return orig(ids)
    s.read_all = read_all
    b = run("ground", servos=s)
    assert getattr(b, "aborted", None) and "overheating" in b.aborted, getattr(b, "aborted", None)
    o = ba.analyse(last("ground"))
    names = [r["phase"] for r in o["rows"]]
    assert names[-1] == "sitdown_after_abort", names


def test_drops():
    s = bl.SimServos(C, drop_per_rad=1.5)
    b = run("drops", "--drop-amps", "0,0.1,0.2", "--caps", "885", "--drop-s", "2", servos=s)
    assert not getattr(b, "aborted", None), b.aborted
    o = ba.analyse(last("drops"))
    r = o["rows"]
    assert r[0]["drop_rate"] == 0 and r[2]["drop_rate"] > r[1]["drop_rate"] > 0, r


def test_stall_trips_and_recovers():
    s = bl.SimServos(C)
    k = 11  # LB femur, push down (+): the floor 0.25 rad below the sleep angle
    s.stops[k] = (C.sleep[k] + 0.25, +1)
    b = run("stall", "--joint", "LB_femur", "--trials", "0.3:0.45,0.7:1.0,1.0:0.4", "--trial-max", "8", "--rest", "1",
            servos=s)
    assert not getattr(b, "aborted", None), b.aborted
    o = ba.analyse(last("stall"))
    out = [r["outcome"] for r in o["rows"]]
    # sim overload rule: |duty| >= 0.5 stalled for 3 s -> trip. 30 % never trips, 70 % trips ~3 s, 40 % never.
    assert out == ["timeout", "trip", "timeout"], o["rows"]
    assert 2.5 < o["rows"][1]["t_trip_s"] < 4.0
    assert s.alarm.sum() == 0


def test_stall_without_obstacle_aborts():
    b = run("stall", "--joint", "LB_femur", "--search", "0.2")
    assert getattr(b, "aborted", None) and "obstacle" in b.aborted


def test_ground_stance_steps_clear_the_floor():
    """Leg steps between STAND and the policy stance: the moving toe stays >= 1.5 cm above its start/end height."""
    import json
    kin = LegKin(C.kin)
    P = Poses(C)
    st = json.load(open(os.path.join(HERE, "policy_stance_2026-11-11.json")))
    stance = C.clip(np.array([st["q_cmd"][n] for n in C.names]))
    for a, z in ((P.STAND, stance), (stance, P.STAND)):
        cur = a.copy()
        for leg in STEP_ORDER:
            segs = sb.leg_step_segments(kin, cur, z, leg)
            zs = kin.toe_positions(np.array([segs[1][0] + (segs[1][1] - segs[1][0]) * s
                                             for s in np.linspace(0, 1, 21)]))[:, leg, 2]
            z0 = min(kin.toe_positions(cur)[leg, 2], kin.toe_positions(z)[leg, 2])
            assert zs.min() >= z0 + 0.015, (leg, zs.min(), z0)
            for q0, q1, _d in segs:
                assert np.all(C.clip(q1) == q1) or np.allclose(C.clip(q1), q1)
            cur = segs[-1][1]
        assert np.allclose(cur, z)


def test_ground_sim_runs_and_stall_monitor():
    s = bl.SimServos(C)
    b = run("ground", "--stand-s", "1", "--stance-s", "1", "--tripod-s", "0.5", servos=s)
    assert not getattr(b, "aborted", None), b.aborted
    o = ba.analyse(last("ground"))
    names = [r["phase"] for r in o["rows"]]
    assert names[0] == "standup" and names[-1] == "sitdown" and "stance_885" in names, names
    m = sb.StallMonitor(619)
    pwm = np.zeros(18)
    pwm[11] = 0.69
    try:
        for j in range(30):
            m(j * 0.02, None, None, pwm)
    except sb.Stall as e:
        assert e.args[0] == 11
        return
    raise AssertionError("stall monitor did not fire")


def test_hung_sequence():
    b = run("hung", "--duties", "0.2,1.0", "--passes", "2", "--amps", "0.1", "--caps", "885", "--hold", "0.4",
            "--freqs", "3,12.5", "--hold-n", "3", "--hold-s", "0.8", "--drop-amps", "0,0.1", "--drop-s", "1")
    assert not getattr(b, "aborted", None), b.aborted
    kinds = sorted(set(os.path.basename(p).split("_", 2)[2].split("_")[0] for p in b.saved))
    assert kinds == ["drops", "hold", "sine", "speed", "step"], kinds
    assert len(b.saved) == 1 + 3 + 3 + 1 + 1


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and sel in n]
    bad = 0
    import io
    import contextlib
    for n, f in tests:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                f()
            print("PASS %s" % n)
        except Exception:
            bad += 1
            print("FAIL %s\n%s\n--- output ---\n%s" % (n, traceback.format_exc(), buf.getvalue()[-3000:]))
    print("%d/%d passed" % (len(tests) - bad, len(tests)))
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
