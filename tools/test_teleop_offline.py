#!/usr/bin/env python3
"""Offline tests for the teleop runner, no robot needed (workstation or Pi, Python 3.8+, NumPy; onnxruntime for the
policy parity tests, which are skipped without it).

    python3 tools/test_teleop_offline.py            # all
    python3 tools/test_teleop_offline.py fsm        # tests whose name contains "fsm"
"""
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import dxl_bus as ax  # noqa: E402
import hexapod_teleop as ht  # noqa: E402
import safety as sf  # noqa: E402
from imu_block import ImuBlock  # noqa: E402
from joint_map import Contract  # noqa: E402
from leg_ik import LegKin  # noqa: E402
from poses import Motion, Poses  # noqa: E402
from proprio import ProprioReader  # noqa: E402

C = Contract()
K = LegKin(C.kin)
P = Poses(C)
DT = C.dt
REF = os.path.join(C.model_dir, "parity_ref.npz")  # policy = HEXAPOD_POLICY (default s1h4)
MAX_STEP = C.rate_limit * DT


def have_ort():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------------------------------ kinematics
def test_fk_ik_roundtrip():
    rng = np.random.RandomState(0)
    n = 0
    for _ in range(2000):
        qf, qt = rng.uniform(C.lo[6], C.hi[6]), rng.uniform(0.0, C.hi[12])  # knee-up branch: tibia > ~-0.3
        r, z = K.plane_fk(qf, qt)
        try:
            qf2, qt2 = K.plane_ik(r, z)
        except ValueError:
            continue
        r2, z2 = K.plane_fk(qf2, qt2)
        assert abs(r2 - r) < 1e-9 and abs(z2 - z) < 1e-9, (r, z, r2, z2)
        if abs(qt - qt2) < 1e-6:
            n += 1
            assert abs(qf - qf2) < 1e-6
    assert n > 1500, n
    r, z = K.legs_plane(C.q_default)
    assert np.allclose(r, 0.0669, atol=5e-4) and np.allclose(z, -0.1703, atol=5e-4), (r, z)
    assert np.abs(K.legs_ik(r, z) - C.q_default).max() < 1e-9
    r, z = K.legs_plane(np.array(C.raw["sleep_positions"]))
    assert np.allclose(r, 0.163, atol=1e-3) and np.allclose(z, -0.030, atol=1e-3), (r, z)


def test_fk_matches_hexasim():
    try:
        import torch
        sys.path.insert(0, "/home/ubuntu/gitrepos/hexasim/source/hexasim")
        from hexasim.rl.s1 import kinematics as hk
    except Exception:
        return "skipped (no torch / hexasim here)"
    rng = np.random.RandomState(1)
    q = rng.uniform(C.lo, C.hi, size=(64, 18))
    ref = hk.toe_positions(torch.tensor(q, dtype=torch.float64)).numpy()
    assert np.abs(K.toe_positions(q) - ref).max() < 1e-9


def test_urdf_mounts_match_vendor_urdf():
    sys.path.insert(0, os.path.join(REPO, "tools", "imu"))
    try:
        import urdf_geom
        legs = {n: (x, y, yaw) for n, x, y, yaw in urdf_geom.legs()}
    except Exception:
        return "skipped (vendor URDF not here)"
    for i, leg in enumerate(C.raw["leg_names"]):
        x, y, yaw = legs[leg]
        assert np.allclose([x, y], C.kin["hip"][i][:2]) and abs(yaw - C.kin["hip_yaw"][i]) < 1e-5, leg


# ------------------------------------------------------------------------------------------------ poses
def _check_motion(segs, start=None):
    m = Motion(segs, DT)
    prev = start
    Q = []
    while True:
        q = m.next()
        if q is None:
            break
        assert np.all(q >= C.lo - 1e-9) and np.all(q <= C.hi + 1e-9), (m.segment.name, q)
        if prev is not None:
            d = np.abs(q - prev).max()
            assert d <= sf.SCRIPT_RATE_MAX * DT + 1e-9, (m.segment.name, d)
        prev = q
        Q.append((m.segment.name, q))
    return Q


def test_poses_in_clip_and_rate():
    Q = _check_motion(P.standup(P.SLEEP + 0.1), start=P.SLEEP + 0.1)
    assert np.abs(Q[-1][1] - C.q_default).max() < 1e-12, "stand-up must end exactly at q_default"
    S = _check_motion(P.sitdown_from_stand(), start=C.q_default)
    assert np.abs(S[-1][1] - P.SLEEP).max() < 1e-9
    total = sum(s.duration for s in P.core)
    assert 8.0 < total < 25.0, total


def test_feet_fixed_and_above_ground():
    """Push-up / rise: every toe keeps its x, y (< 2 mm) and all toes share z (body moves straight up). Steps: stance
    toes stay at the ground height, the swing toe never goes below it."""
    for seg in P.core:
        s = np.linspace(0, 1, 101)
        T = np.array([K.toe_positions(seg.q(x)) for x in s])  # (101, 6, 3)
        if seg.name in ("push_up", "rise"):
            assert np.abs(T[:, :, :2] - T[0, :, :2]).max() < 2e-3, seg.name
            assert np.abs(T[:, :, 2] - T[:, :1, 2]).max() < 1e-9, seg.name
        if seg.name.startswith("step_"):
            leg = C.raw["leg_names"].index(seg.name[5:])
            ground = T[0, :, 2].min()
            stance = [i for i in range(6) if i != leg]
            assert np.abs(T[:, stance, :] - T[0, stance, :]).max() < 1e-9, seg.name
            assert T[:, leg, 2].min() >= ground - 1e-9, seg.name
            assert T[:, leg, 2].max() > ground + 0.015, seg.name


def test_belly_and_heights():
    """Belly height = -(toe z) - 0.033 when standing on the toes."""
    for name, q, belly in (("stand", P.STAND, 0.137), ("wide_stand", P.WIDE_STAND, 0.087)):
        z = K.toe_positions(q)[:, 2]
        assert np.allclose(z, z[0]) and abs(-z[0] - 0.033 - belly) < 2e-3, (name, z)
    assert np.allclose(K.toe_positions(P.PLANT)[:, 2], -0.033)


def test_sit_reverses_any_point_of_standup():
    for k in range(len(P.core)):
        mid = P.core[k].q(0.5)
        segs = P.sitdown_from_core(k, mid)
        Q = _check_motion(segs, start=mid)
        assert np.abs(Q[-1][1] - P.SLEEP).max() < 1e-9, k


# ------------------------------------------------------------------------------------------------ proprio + policy
def _run_reader(ref):
    r = ProprioReader(18, DT, C.tick)
    out = {"q": [], "qd": [], "mask": []}
    for k in range(len(ref["q_true"])):
        if k == 0:
            r.reset(ref["q_true"][0])
        q = np.where(ref["valid"][k], ref["q_true"][k], 123.0)  # garbage where dropped
        r.update(q, ref["valid"][k])
        for key in out:
            out[key].append(getattr(r, key).copy())
    return {k: np.array(v) for k, v in out.items()}


def test_proprio_parity():
    ref = np.load(REF)
    out = _run_reader(ref)
    for k in ("q", "qd", "mask"):
        err = np.abs(out[k] - ref[k]).max()
        assert err < 1e-5 if k == "qd" else err < 1e-6, (k, err)  # torch float32 vs float64
    assert (~ref["valid"]).sum() > 50


def test_policy_parity():
    if not have_ort():
        return "skipped (no onnxruntime)"
    from s1h_policy import Clock, ObsBuilder, S1HPolicy, slew
    ref = np.load(REF)
    pol, ob, ck = S1HPolicy(C), ObsBuilder(C), Clock(C)
    a_prev, qcmd_prev = np.zeros(18), C.q_default.copy()
    worst_op = worst_a = 0.0
    for k in range(len(ref["action"])):
        # feed the reference reader output (float32 in torch); the reader itself is tested above
        o_p = ob.frame(ref["q"][k], ref["qd"][k], ref["mask"][k], qcmd_prev, a_prev, ref["imu9"][k],
                       ck.frame(ref["cmd"][k]))
        worst_op = max(worst_op, np.abs(o_p - ref["o_p"][k]).max())
        a = pol.act(o_p, ref["cmd"][k])
        worst_a = max(worst_a, np.abs(a - ref["action"][k]).max())
        q_cmd = pol.q_cmd(a) if MAX_STEP <= 0 else slew(pol.q_cmd(a), qcmd_prev, MAX_STEP)
        assert np.abs(q_cmd - ref["q_cmd"][k]).max() < 1e-4
        a_prev, qcmd_prev = a, q_cmd
    assert worst_op < 1e-5, worst_op
    assert worst_a < 1e-4, worst_a
    io = np.load(os.path.join(C.model_dir, "fused_io.npz"))
    a = pol.sess.run(["action"], {"hist": io["hist"], "obs_base": io["obs_base"]})[0]
    assert np.abs(a - io["action"]).max() < 1e-5
    return "max |o_p| err %.1e, max |a| err %.1e over %d steps" % (worst_op, worst_a, len(ref["action"]))


def test_policy_end_to_end_with_numpy_reader():
    if not have_ort():
        return "skipped (no onnxruntime)"
    from s1h_policy import Clock, ObsBuilder, S1HPolicy, slew
    ref = np.load(REF)
    rd = _run_reader(ref)
    pol, ob, ck = S1HPolicy(C), ObsBuilder(C), Clock(C)
    a_prev, qcmd_prev = np.zeros(18), C.q_default.copy()
    worst = 0.0
    for k in range(len(ref["action"])):
        o_p = ob.frame(rd["q"][k], rd["qd"][k], rd["mask"][k], qcmd_prev, a_prev, ref["imu9"][k],
                       ck.frame(ref["cmd"][k]))
        a = pol.act(o_p, ref["cmd"][k])
        worst = max(worst, np.abs(a - ref["action"][k]).max())
        a_prev, qcmd_prev = a, pol.q_cmd(a) if MAX_STEP <= 0 else slew(pol.q_cmd(a), qcmd_prev, MAX_STEP)
    assert worst < 1e-4, worst
    return "max |a| err %.1e" % worst


# ------------------------------------------------------------------------------------------------ state machine
class StubPolicy:
    def __init__(self):
        self.nan = False
        self.n = 0

    def reset(self):
        self.n = 0

    def act(self, o_p, cmd):
        self.n += 1
        assert o_p.shape == (111,) and np.all(np.isfinite(o_p))
        return np.full(18, np.nan) if self.nan else 0.02 * np.sin(0.1 * self.n + np.arange(18))

    def q_cmd(self, a):
        return np.clip(C.q_default + C.scale * a, C.clip_lo, C.clip_hi)


class Sim:
    """Runner on a FakeBus + FakeImu with a virtual clock."""

    def __init__(self, policy=None, q0=None, imu_thread=False):
        self.t = 100.0
        q0 = C.sleep if q0 is None else q0
        self.bus = ax.FakeBus({int(i): float(q) for i, q in zip(C.ids, q0)}, speed=6.0, dt=DT)
        self.blk = ImuBlock(os.path.join(REPO, "config", "imu_calib.json"))
        self.g = np.array([0.0, 0.0, -1.0])
        self.imu = ht.FakeImu(self.blk, tilt=lambda: self.g)
        self.blk.start([self.imu.read() for _ in range(50)])
        self.r = ht.Runner(C, self.bus, self.imu, self.blk, policy or StubPolicy(), clock=lambda: self.t, out=None,
                           imu_thread=imu_thread)
        self.states = []
        self.jump = 0.0

    def tick(self, keys=(), n=1):
        for _ in range(n):
            self.t += DT + self.jump
            self.bus.advance()
            self.r.tick(keys)
            keys = ()
            self.states.append(self.r.state)
        return self.r.state

    def until(self, state, max_s=60.0):
        for _ in range(int(max_s / DT)):
            if self.r.state == state:
                return True
            self.tick()
        return self.r.state == state

    def stand(self):
        self.tick(n=3)  # first reads (the runner refuses to stand before all 18 servos have been read)
        self.tick(["u"])
        assert self.r.state == ht.STANDUP, self.r.state
        assert self.until(ht.STAND), self.r.state
        assert all(self.bus.on.values()) and self.bus.pwm[1] == ht.TORQUE_SCRIPT

    def walk(self):
        self.stand()
        self.tick(["p"])
        assert self.r.state == ht.POLICY and self.bus.pwm[1] == ht.TORQUE_POLICY
        self.tick(n=10)


def _ends_off(sim, max_s=60.0, expect_freeze=True):
    assert sim.until(ht.OFF, max_s), sim.r.state
    assert not any(sim.bus.on.values()), "torque must be off"
    assert ht.SIT in sim.states, "must sit down first"
    q = np.array([sim.bus.q[int(i)] for i in C.ids])
    assert np.abs(q - P.SLEEP).max() < 0.02, np.abs(q - P.SLEEP).max()


def test_fsm_standup_sit_cycle():
    s = Sim()
    s.stand()
    q = np.array([s.bus.q[int(i)] for i in C.ids])
    assert np.abs(q - C.q_default).max() < 0.01
    s.tick(["j"])
    _ends_off(s, expect_freeze=False)


def test_fsm_policy_and_recover():
    s = Sim()
    s.walk()
    s.tick(["w"])
    s.tick(n=15)
    assert s.r.cmd.applied[0] == 0.05, s.r.cmd.applied
    s.tick(["p"])
    assert s.r.state == ht.RECOVER and s.bus.pwm[1] == ht.TORQUE_SCRIPT
    assert s.until(ht.STAND)
    s.tick(["j"])
    _ends_off(s)


def test_fsm_fault_drop_burst():
    s = Sim()
    s.walk()
    s.bus.drop_fn = lambda step, i: i == 5 and step < s.bus.step_n + 8 or False
    start = s.bus.step_n
    s.bus.drop_fn = lambda step, i: i == 5 and step < start + 8
    s.tick(n=8)
    assert s.r.state == ht.SIT and "consecutive" in s.r.fault_log[-1], s.r.fault_log
    _ends_off(s)


def test_fsm_fault_overrun():
    s = Sim()
    s.walk()
    s.jump = 0.012  # 32 ms periods
    s.tick(n=6)
    s.jump = 0.0
    assert s.r.state == ht.SIT and "overrun" in s.r.fault_log[-1], s.r.fault_log
    _ends_off(s)


def test_fsm_fault_tilt():
    s = Sim()
    s.walk()
    a = np.radians(40)
    s.g = np.array([np.sin(a), 0.0, -np.cos(a)])
    s.tick(n=150)  # complementary filter time constant ~1 s
    assert ht.SIT in s.states and any("tilt" in f for f in s.r.fault_log), s.r.fault_log
    s.g = np.array([0.0, 0.0, -1.0])
    _ends_off(s)


def test_fsm_fault_nan_policy():
    pol = StubPolicy()
    s = Sim(pol)
    s.walk()
    pol.nan = True
    s.tick()
    assert s.r.state == ht.SIT and "not finite" in s.r.fault_log[-1], s.r.fault_log
    _ends_off(s)


def test_fsm_sighup():
    s = Sim()
    s.walk()
    s.r.pending_exit = "SIGHUP"
    s.tick()
    assert s.r.state == ht.SIT
    assert s.until(ht.EXIT, 60.0), s.r.state
    assert not any(s.bus.on.values()) and ht.SIT in s.states


def test_fsm_low_voltage():
    s = Sim()
    s.walk()
    s.bus.volts = 10.3
    s.tick(n=sf.V_WINDOW)  # health every 2nd step, median over 18 reads: ~10 low readings (~20 steps) to trip
    assert s.r.state == ht.POLICY, "a few low readings must not trip"
    s.tick(n=4)
    assert s.r.state == ht.SIT and "battery" in s.r.fault_log[-1], s.r.fault_log
    _ends_off(s)
    s.tick(["u"])
    assert s.r.state == ht.OFF, "stand-up must be refused while the battery is low"


def test_fsm_imu_stale():
    s = Sim()
    s.walk()
    s.imu.fail = True
    s.tick(n=5)
    assert s.r.state == ht.SIT and "IMU" in s.r.fault_log[-1], s.r.fault_log
    s.imu.fail = False
    _ends_off(s)


def test_fsm_imu_thread():
    s = Sim(imu_thread=True)
    s.walk()
    s.tick(n=50)
    assert s.r.state == ht.POLICY and s.r.last_level[0] < 2, s.r.last_level
    s.imu.fail = True  # an OSError inside the worker reaches the runner like a direct read
    s.tick(n=5)
    assert s.r.state == ht.SIT and "IMU" in s.r.fault_log[-1], s.r.fault_log
    s.imu.fail = False
    _ends_off(s)


def test_fsm_imu_thread_stuck_read():
    import threading
    s = Sim(imu_thread=True)
    s.walk()
    gate = threading.Event()
    read = s.imu.read
    s.imu.read = lambda: (gate.wait(), read())[1]  # I2C read hangs
    s.tick(n=5)
    assert s.r.state == ht.SIT and "IMU" in s.r.fault_log[-1], s.r.fault_log
    gate.set()
    s.imu.read = read
    _ends_off(s)


class ChainBus(ax.DxlBus):
    """DxlBus with the SYNC_READ chain modelled: an id in `silent` does not answer, nor does any id behind it in the
    list (each servo waits for the previous one's reply). No serial port."""

    def __init__(self):
        self.timeout, self.sync_timeout = 0.003, 0.010
        self.last_error, self.n_tx, self.n_drop, self.n_retry = {}, 0, 0, 0
        self.order, self.breaks = None, {}
        self.silent, self.lists = set(), []

    def sync_read(self, ids, addr, n):
        self.lists.append(list(ids))
        out = {}
        for i in ids:
            if i in self.silent:
                break
            out[i] = 2048 + i
        self.last_sync = (15 * len(out), len(out), 0)
        self.n_drop += len(ids) - len(out)
        return out


def test_read_order_moves_chain_breaker_last():
    b = ChainBus()
    ids = [int(i) for i in C.ids]
    culprit = ids[15]  # LF tibia (id 9): the 2026-11-11 culprit, 3rd from the end in sim order
    b.silent = {culprit}
    first = b.read_positions(ids)
    assert b.last_read_diag[1] == 15, "chain model: 1st SYNC_READ loses the culprit + the 2 ids behind it"
    assert [k for k, v in enumerate(first) if v is None] == [15], "the retry asks the culprit last"
    assert b.order[-1] == culprit and b.last_read_diag[4] == culprit
    for _ in range(5):
        q = b.read_positions(ids)
        assert b.last_read_diag[1] == 17 and b.lists[-1] == [culprit], (b.last_read_diag, b.lists[-1])
        assert [k for k, v in enumerate(q) if v is None] == [15], q
    assert all(abs(v - ax.pos_to_rad(2048 + i)) < 1e-12 for v, i in zip(q, ids) if v is not None)
    b.silent = {ids[2]}  # a second breaker: goes before the first, which stays last
    for _ in range(3):
        b.read_positions(ids)
    assert b.order[-2:] == [ids[2], culprit], b.order[-3:]
    b.silent = set()
    assert all(v is not None for v in b.read_positions(ids))
    return "drops per break 3 -> 1"


def test_fsm_fault_during_standup_reverses():
    s = Sim()
    s.tick(n=3)
    s.tick(["u"])
    s.tick(n=int(6.0 / DT))  # into the push-up
    k = s.r.core_k
    assert k is not None and k >= 1, k
    s.r.react(sf.FAULT, "test")
    names = [seg.name for seg in s.r.motion.segs]
    assert names[0] == "freeze" and names[1].startswith("back_") and "to_stand" not in names, names
    _ends_off(s)


def _pin(sim, k, offset):
    """Block joint k (sim index) at its present angle + offset: the FakeBus servo no longer follows its goal."""
    i = int(C.ids[k])
    q_pin = sim.bus.q[i] + offset
    adv = sim.bus.advance

    def advance():
        adv()
        sim.bus.q[i] = q_pin
    sim.bus.advance = advance
    return lambda: setattr(sim.bus, "advance", adv)


def test_fsm_stall_fast_tier():
    s = Sim()
    s.stand()
    assert s.r.pwm_cap == ht.TORQUE_SCRIPT and s.r.safety.duty.max() < 0.1, s.r.safety.duty.max()
    release = _pin(s, 7, 0.13)  # RM femur blocked 0.13 rad off: P law asks 79 %, the 70 % cap holds it
    n = 0
    while s.r.state == ht.STAND and n < 200:
        s.tick()
        n += 1
        if n == int(0.75 * sf.STALL_FAST_S / DT):
            assert s.r.last_level[0] == sf.WARN and "near stall" in s.r.last_level[1], s.r.last_level
    assert s.r.state == ht.SIT and "stall" in s.r.fault_log[-1] and "idx 7" in s.r.fault_log[-1], s.r.fault_log
    assert abs(n * DT - sf.STALL_FAST_S) <= 3 * DT, n * DT  # far inside the ~4 s overload trip
    release()
    _ends_off(s)


def test_fsm_stall_slow_tier_and_no_false_trip():
    s = Sim()
    s.stand()
    release = _pin(s, 13, 0.065)  # RM tibia: 40 % duty, below both tiers -> never a fault
    s.tick(n=int(8.0 / DT))
    assert s.r.state == ht.STAND and not s.r.fault_log, (s.r.state, s.r.fault_log)
    assert abs(s.r.safety.duty[13] - 0.065 / ht.sat_err(ht.KP_AX_MATCH)) < 0.04, s.r.safety.duty[13]
    release()
    s.tick(n=5)
    release = _pin(s, 13, 0.085)  # 52 %: the slow tier only
    n = 0
    while s.r.state == ht.STAND and n < 400:
        s.tick()
        n += 1
    assert s.r.state == ht.SIT and ">= 45 %" in s.r.fault_log[-1], s.r.fault_log
    assert abs(n * DT - sf.STALL_SLOW_S) <= 3 * DT, n * DT
    release()
    _ends_off(s)


def test_stall_monitor_unit():
    m = sf.Safety(18, DT)
    m.imu_ok(0.0)
    m.sat_err = ht.sat_err(ht.KP_AX_MATCH)
    assert abs(m.sat_err - 0.1638) < 1e-3, m.sat_err
    q, g, ok = np.zeros(18), np.array([0, 0, -1.0]), np.ones(18, bool)
    cmd = np.zeros(18)
    cmd[3] = 0.5  # asks 305 %: the estimate is clipped at the cap

    def run(steps, cap, valid=ok, c=cmd, t0=0.0):
        lvl = None
        for k in range(steps):
            m.imu_ok(t0 + k * DT)
            lvl = m.step(t0 + k * DT, valid, q, c, g, DT, True, cap=cap)
        return lvl
    assert run(1, 1.0)[0] == sf.OK and abs(m.duty[3] - 1.0) < 1e-9
    # a 40 % cap is below both tiers (30 % never tripped on the bench): no stall fault however long
    m.stall_t[:] = 0
    lvl = run(600, 0.40)
    assert "stall" not in lvl[1], lvl
    # dropped reads neither advance nor reset the timer
    m.stall_t[:] = 0
    drop = ok.copy()
    drop[3] = False
    run(20, 1.0)
    held = m.stall_t[0][3]
    run(100, 1.0, valid=drop)
    assert m.stall_t[0][3] == held, (m.stall_t[0][3], held)
    # the error going away resets it; torque off resets it; no sat_err = monitor off
    run(1, 1.0, c=np.zeros(18))
    assert m.stall_t.max() == 0
    run(20, 1.0)
    m.step(0.0, ok, q, cmd, g, DT, False, cap=1.0)
    assert m.stall_t.max() == 0 and m.duty.max() == 0
    m.sat_err = None
    assert "stall" not in run(200, 1.0)[1]


def test_setup_refuses_latched_alarm():
    bus = ax.FakeBus({int(i): float(q) for i, q in zip(C.ids, C.sleep)})
    bus.hw_error = lambda i: 0x20 if i == 2 else 0
    try:
        ht.setup_servos(bus, C, lambda m: None)
        raise AssertionError("a latched overload must refuse the start")
    except SystemExit as e:
        assert "LB femur" in str(e) and "overload" in str(e) and "--clear" in str(e), str(e)
    assert ax.POS_P not in bus.regs[1] or bus.regs[1][ax.POS_P] != ht.KP_AX_MATCH  # refused before any RAM write
    bus.hw_error = lambda i: None if i == 9 else 0
    try:
        ht.setup_servos(bus, C, lambda m: None)
        raise AssertionError("an unreadable alarm register must refuse the start")
    except SystemExit as e:
        assert "not readable" in str(e), str(e)
    bus.hw_error = lambda i: 0x80 if i == 4 else 0  # a bit outside ALARM_MASK is not a latched shutdown
    ht.setup_servos(bus, C, lambda m: None)


def test_fsm_bus_dead_kills():
    s = Sim()
    s.walk()
    s.bus.dead = True
    s.tick(n=sf.BUS_DEAD_STEPS + 1)
    assert s.r.state == ht.EXIT and not any(s.bus.on.values())


def test_fsm_x_and_freeze():
    s = Sim()
    s.walk()
    s.tick(["f"])
    assert s.r.state == ht.FROZEN
    s.tick(n=10)
    s.tick(["x"])
    assert s.r.state == ht.OFF and not any(s.bus.on.values())


def test_fsm_relaxed_lying_pose_accepted():
    """The pose measured on the robot 2026-11-06 (belly down, torque off): u and c must start without a confirm."""
    q = C.sleep.copy()
    q[6:12] += [0.10, 0.0, 0.13, 0.37, 0.16, 0.12]  # femurs RF RM RB LF LM LB
    q[12:18] += [-0.39, -0.42, -0.39, -0.85, -0.39, -0.41]  # tibias
    s = Sim(q0=q)
    s.tick(n=2)
    s.tick(["u"])
    assert s.r.state == ht.STANDUP and s.r.motion.segs[0].duration < 4.0, s.r.state
    s2 = Sim(q0=q)
    s2.tick(n=2)
    s2.tick(["c"])
    assert s2.r.state == ht.SIGNCHECK, s2.r.state


def test_fsm_far_pose_needs_confirm():
    q = C.sleep.copy()
    q[8] += 0.9  # RB femur well beyond the 0.6 rad tolerance
    s = Sim(q0=q)
    s.tick(n=2)
    s.tick(["u"])
    assert s.r.state == ht.OFF
    s.tick(["u"])
    assert s.r.state == ht.STANDUP and s.r.motion.segs[0].duration >= 4.0


def test_fsm_signcheck():
    s = Sim()
    s.tick(n=2)
    s.tick(["c"])
    assert s.r.state == ht.SIGNCHECK
    for _ in range(18):
        for _ in range(400):
            s.tick()
            if s.r.sign["await"]:
                break
        assert s.r.sign["await"]
        s.tick(["y"])
    assert s.until(ht.OFF) and len(s.r.sign["answers"]) == 18


def test_command_snapping():
    cmd = ht.Command(C, 0.10, 0.2)
    assert cmd.vx_steps == [0.0, 0.05, 0.075, 0.1], cmd.vx_steps
    cmd.key("a")
    assert cmd.target[1] == 0.0, "v_y only when walking"
    cmd.key("q")
    assert np.allclose(cmd.target, [0, 0, 0.1]), cmd.target  # turn on the spot
    cmd.key("q")
    cmd.key("q")
    assert np.allclose(cmd.target, [0, 0, 0.2])  # capped
    cmd.key("w")
    cmd.key("a")
    cmd.key("a")
    cmd.key("a")
    assert np.allclose(cmd.target, [0.05, 0.05, 0.2]), cmd.target
    for _ in range(9):
        cmd.key("w")
    assert cmd.target[0] == 0.10
    cmd.key(" ")
    assert np.allclose(cmd.target, 0)
    wide = ht.Command(C, 0.2, 0.4)
    assert wide.vx_steps[-1] == 0.2 and wide.wz_max == 0.4
    for v in ([0.03, 0.04, 0.0], [0.0, 0.0, 0.05], [0.3, 0.1, 0.9]):
        t = wide.snap(*v)
        standing = np.allclose(t, 0)
        turn = t[0] == 0 and t[1] == 0 and 0.1 <= abs(t[2]) <= 0.4
        walk = 0.05 <= t[0] <= 0.2 and abs(t[1]) <= 0.05 and abs(t[2]) <= 0.4
        assert standing or turn or walk, (v, t)
    # 0.2 s gate
    g = ht.Command(C, 0.1, 0.2)
    g.key("w")
    assert g.update(0.0)[0] == 0.05
    g.key("w")
    assert g.update(0.1)[0] == 0.05 and g.update(0.21)[0] == 0.075


def test_command_fine_ramp():
    c = ht.Command(C, 0.10, 0.2)
    c.fine(walk_lo=0.02, ramp_s=1.0)
    assert c.vx_steps == [0.0, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1], c.vx_steps
    assert c.update(0.0)[0] == 0.0
    c.key("w")
    assert abs(c.update(0.02)[0] - 0.02) < 1e-9, "walking starts at the slowest step, not from 0"
    for _ in range(5):
        c.key("w")  # target 0.10
    v, t = [], 0.02
    while t < 3.0:
        t += 0.02
        v.append(c.update(t)[0])
    dv = np.diff([0.02] + v)
    assert dv.max() <= 0.05 * 0.02 / 1.0 + 1e-9 and abs(v[-1] - 0.10) < 1e-9, (dv.max(), v[-1])
    c.key(" ")
    assert np.allclose(c.update(t + 0.02), 0), "stop is immediate"
    c.key("w")
    c.key("a")
    assert c.target[1] > 0, "v_y allowed from walk_lo"


def test_retry_puts_silent_id_last():
    bus = ax.DxlBus.__new__(ax.DxlBus)
    bus.n_tx = bus.n_drop = bus.n_retry = 0
    bus.order, bus.breaks = None, {}
    bus.sync_timeout = 0.01
    asked = []

    def fake_sync_read(ids, addr, n):  # id 16 silent: it and every id after it in the request drop
        asked.append(list(ids))
        out = {}
        for i in ids:
            if i == 16:
                break
            out[i] = 2048
        bus.last_sync = (15 * len(out), len(out), 0)
        return out
    bus.sync_read = fake_sync_read
    q = bus.read_positions([14, 15, 16, 17, 18])
    assert asked[1] == [17, 18, 16], asked
    assert q[3] is not None and q[4] is not None and q[2] is None, q
    assert bus.last_read_diag[2] == 2, bus.last_read_diag

def test_bus_packets():
    # e-manual Protocol 2.0 ping example for ID 1
    assert ax.packet(1, ax.PING) == bytes.fromhex("FFFFFD0001030001194E"), ax.packet(1, ax.PING).hex()
    assert ax.rad_to_pos(0.0) == 2048 and abs(ax.pos_to_rad(ax.rad_to_pos(0.3)) - 0.3) <= ax.TICK / 2
    # byte stuffing in params
    assert ax._stuff(b"\xff\xff\xfd\x01") == b"\xff\xff\xfd\xfd\x01"
    # a status stream: 3 position replies, one corrupted, plus leading noise
    import struct

    def status(i, val):
        body = bytes([0xFF, 0xFF, 0xFD, 0x00, i]) + struct.pack("<H", 8) + bytes([0x55, 0]) + struct.pack("<i", val)
        return body + struct.pack("<H", ax.crc16(body))
    bad = bytearray(status(2, 1500))
    bad[-3] ^= 0x01
    got = ax.parse_status(b"\x00\x13" + status(1, 2048) + bytes(bad) + status(3, 4095), 4)
    assert set(got) == {1, 3} and struct.unpack("<i", got[3][1])[0] == 4095, got
    try:
        ax.FakeBus({1: 0.0}).write(1, ax.MIN_POSITION, 0, 4)
        raise AssertionError("EEPROM write must be refused")
    except ax.BusError:
        pass


def test_health_full_stuffed_reply():
    """The 23-byte health block has signed fields; a reply containing FF FF FD arrives byte-stuffed and must parse."""
    import struct
    pwm, load, vel = -1, -3, -2  # FF FF | FD FF | FE FF FF FF: PWM -1 then load low byte FD -> stuffing
    params = struct.pack("<hhiiIIHB", pwm, load, vel, 2048, 0, 0, 115, 41)
    assert b"\xff\xff\xfd" in params
    stuffed = params.replace(b"\xff\xff\xfd", b"\xff\xff\xfd\xfd")
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, 7]) + struct.pack("<H", len(stuffed) + 4) + bytes([0x55, 0]) + stuffed
    pk = body + struct.pack("<H", ax.crc16(body))
    got = ax.parse_status_stuffed(b"\x00" + pk)
    assert set(got) == {7} and got[7][1] == params, got
    v, T, d, l, w = ax.unpack_health(got[7][1])
    assert (v, T) == (11.5, 41.0) and abs(d + 1 / 885.0) < 1e-12 and abs(l + 0.003) < 1e-12
    assert abs(w + 2 * ax.VEL_UNIT) < 1e-12
    assert ax.parse_status(pk, 23) == {}  # the fixed-length parser would have dropped it


def test_health_pwm_logged():
    """Each health read logs the joint index, Present PWM, load and velocity next to the voltage."""
    s = Sim()
    s.walk()
    s.tick(n=60)
    rows = s.r.log.rows
    hk, hp, hv = (np.array(rows[k], dtype=float) for k in ("hk", "hp", "hv"))
    on = np.isfinite(hk)
    assert on.sum() >= 30 and np.isfinite(hp[on]).all() and np.isfinite(hv[on]).all()
    assert np.isnan(hp[~on]).all() and (np.abs(hp[on]) <= 1.0).all()
    assert set(hk[on].astype(int)) == set(range(18)), sorted(set(hk[on].astype(int)))
    assert np.abs(hp[on]).max() > 0.0  # the policy moves the legs: some joint has a nonzero duty
    for k in ("hl", "hw"):
        assert np.isfinite(np.array(rows[k], dtype=float)[on]).all(), k


def test_proprio_tick_is_sim_tick():
    """The reader quantises to the tick of the servo model the policy trained on (AX-12A 10-bit or XL430 12-bit)."""
    ticks = {"ax12a": np.radians(300.0) / 1023.0, "xl430": 2.0 * np.pi / 4096.0}
    assert abs(C.tick - ticks[C.servo.get("model", "ax12a")]) < 1e-15, C.tick
    s = Sim()
    assert s.r.reader.tick == C.tick and s.r.rate_limit == C.rate_limit


def test_setup_servos_fake():
    bus = ax.FakeBus({int(i): float(q) for i, q in zip(C.ids, C.sleep)})
    ht.setup_servos(bus, C, lambda m: None)
    r = bus.regs[1]
    assert r[ax.POS_P] == ht.KP_AX_MATCH and r[ax.POS_D] == 0 and r[ax.POS_I] == 0 and r[ax.PROFILE_VELOCITY] == 0
    assert r[ax.GOAL_PWM] == ht.TORQUE_SCRIPT
    bus.regs[5][ax.OPERATING_MODE] = 4
    try:
        ht.setup_servos(bus, C, lambda m: None)
        raise AssertionError("wrong operating mode must abort")
    except SystemExit:
        pass


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: 0) if n.startswith("test_") and sel in n]
    fails = 0
    for n, f in tests:
        try:
            note = f()
            print("PASS %-45s %s" % (n, note or ""))
        except Exception:
            fails += 1
            print("FAIL %s\n%s" % (n, traceback.format_exc()))
    print("%d/%d passed" % (len(tests) - fails, len(tests)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
