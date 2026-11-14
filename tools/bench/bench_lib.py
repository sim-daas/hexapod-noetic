"""Servo bench library for the XL430-W250 motor model (Pi, Python 3.8, NumPy; no ROS, no vendor SDK).

Two interchangeable servo back-ends with one API:
  RealServos(DxlBus)   the robot (Protocol 2.0 through scripts/dxl_bus.py)
  SimServos(...)       an XL430-like DC-motor simulation on a virtual clock, for the offline tests

What the bench writes beyond the runner (scripts/dxl_bus.py writes RAM only):
  - Operating Mode (EEPROM addr 11) of ONE servo: 3 (position) <-> 16 (PWM), only with its torque off. Every switch
    is recorded in PWM_MARK first, and every bench start restores 3 on the ids listed there (Ctrl-C, crash, power loss).
    The teleop runner refuses to start on a servo whose mode is not 3, so a forgotten switch cannot reach the policy.
  - REBOOT (instruction 0x08) to clear a latched hardware alarm (overload). A reboot resets RAM (torque off, gains back
    to factory P 640 / D 3600, Goal PWM = PWM Limit): the bench re-applies its gains afterwards.

Units in every record: q rad (2048 = 0), vel rad/s (Present Velocity x 0.229 rpm), pwm and load as fractions
(Present PWM / 885, Present Load / 1000; signed, + = increasing position), V volts, T deg C.
"""
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import dxl_bus as dx  # noqa: E402
from joint_map import Contract  # noqa: E402

PWM_FULL = 885
KP = 1061  # runner gain (AX-12A slope-32 match: full PWM at 0.1638 rad)
CAP_SCRIPT = int(0.70 * PWM_FULL)  # 619, the runner's Goal PWM for scripted moves
SAT_ERR = PWM_FULL / (KP / 128.0) * dx.TICK  # 0.1638 rad: position error at which P 1061 asks 100 % PWM
VEL_UNIT = 0.229 * 2.0 * math.pi / 60.0  # rad/s per Present Velocity unit
OP_PWM = 16
REBOOT = 0x08
BLOCK_ADDR, BLOCK_N = dx.PRESENT_PWM, 23  # 124..146: PWM, Load, Velocity, Position, VelTraj, PosTraj, Voltage, Temp
ALL_ADDR, ALL_N = dx.PRESENT_PWM, 12  # 124..135 for SYNC_READ of all 18

TEMP_ABORT = 65.0  # deg C (servo shutdown limit 72)
V_ABORT = 10.0  # V, any reading below aborts a test (battery protection)
V_LOW_WARN = 11.0

LOGDIR = os.path.expanduser("~/hexapod_logs/bench")
PWM_MARK = os.path.join(LOGDIR, ".pwm_mode_ids.json")


class Abort(Exception):
    pass


def unstuff(p):
    return p.replace(b"\xff\xff\xfd\xfd", b"\xff\xff\xfd")


def parse_status_any(buf):
    """-> {id: (error, params)} for every valid status packet in buf. Unlike dxl_bus.parse_status this takes the
    length from the header and removes byte stuffing: the bench reads signed PWM/load/velocity, where e.g. load -1
    (FF FF) followed by a velocity byte FD makes the servo stuff an extra FD."""
    out, i = {}, 0
    while i + 11 <= len(buf):
        if buf[i:i + 4] != b"\xff\xff\xfd\x00":
            i += 1
            continue
        L = struct.unpack("<H", buf[i + 5:i + 7])[0]
        end = i + 7 + L
        if L < 4 or end > len(buf):
            i += 1
            continue
        pk = buf[i:end]
        if pk[7] == 0x55 and dx.crc16(pk[:-2]) == struct.unpack("<H", pk[-2:])[0]:
            out[pk[4]] = (pk[8], unstuff(bytes(pk[9:-2])))
            i = end
        else:
            i += 1
    return out


def le16s(v):
    return int(v) & 0xFFFF


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "?"


# ====================================================================================================== real servos
class RealServos:
    """Thin layer over dxl_bus.DxlBus with the bench's extra reads/writes."""

    sim = False

    def __init__(self, bus):
        self.bus = bus
        self.now = time.monotonic
        self.sleep = time.sleep

    # ---- reads
    def _read_raw(self, i, addr, n):
        b = self.bus
        b.n_tx += 1
        b._send(dx.packet(i, dx.READ, struct.pack("<HH", addr, n)))
        buf = b._recv(11 + n, b.timeout)
        if len(buf) >= 7 and buf[:4] == b"\xff\xff\xfd\x00":
            need = 7 + struct.unpack("<H", buf[5:7])[0] - len(buf)  # stuffed reply: a few bytes longer
            if 0 < need <= 8:
                buf += b._recv(need, b.timeout)
        r = parse_status_any(buf).get(i)
        if r is not None and len(r[1]) != n:
            r = None
        if r is None:
            b.ser.reset_input_buffer()
            return None
        if r[0] & dx.ALERT:
            b.last_error[i] = r[0]
        return r

    def block(self, i):
        """One READ of 124..146 -> dict or None (missing reply)."""
        t = self.now()
        r = self._read_raw(i, BLOCK_ADDR, BLOCK_N)
        if r is None:
            return None
        err, p = r
        pwm, load, vel, pos, _vt, _pt, volt, temp = struct.unpack("<hhiiIIHB", p)
        return {"t": t, "pwm": pwm / float(PWM_FULL), "load": load / 1000.0, "vel": vel * VEL_UNIT,
                "q": dx.pos_to_rad(pos), "V": volt / 10.0, "T": float(temp), "alert": bool(err & dx.ALERT)}

    def read_all(self, ids):
        """SYNC_READ 124..135 of all ids -> {id: (pwm, load, vel, q)} for the ids that answered, + diag."""
        b = self.bus
        order = b.read_order(list(ids))
        b._send(dx.packet(dx.BROADCAST, dx.SYNC_READ, struct.pack("<HH", ALL_ADDR, ALL_N) + bytes(order)))
        # slack for stuffed replies (the read then ends on the 2 ms gap instead of the byte count)
        raw = b._recv_gap((11 + ALL_N) * len(order) + 4 * len(order), b.sync_timeout, dx.GAP_S)
        got = {k: v for k, v in parse_status_any(raw).items() if len(v[1]) == ALL_N}
        out = {}
        for i in order:
            r = got.get(i)
            if r is None:
                continue
            if r[0] & dx.ALERT:
                b.last_error[i] = r[0]
            pwm, load, vel, pos = struct.unpack("<hhii", r[1])
            if 0 <= pos <= 4095:
                out[i] = (pwm / float(PWM_FULL), load / 1000.0, vel * VEL_UNIT, dx.pos_to_rad(pos))
        missing = [i for i in order if i not in out]
        if missing:
            b.ser.reset_input_buffer()
            b._chain_broke(missing[0])
        return out, (missing[0] if missing else 0)

    def read_positions(self, ids):
        return self.bus.read_positions(list(ids))

    def health(self, i):
        return self.bus.health(i)

    def read(self, i, addr, n):
        return self.bus.read(i, addr, n)

    def ping(self, i):
        return self.bus.ping(i)

    def hw_error(self, i):
        return self.bus.hw_error(i)

    def take_alerts(self):
        e = dict(self.bus.last_error)
        self.bus.last_error.clear()
        return e

    # ---- writes (RAM)
    def goals(self, ids, qs):
        self.bus.write_goals([int(i) for i in ids], [float(q) for q in qs])

    def duty(self, i, d):
        """PWM mode: Goal PWM = d x 885 (signed)."""
        v = int(round(max(-1.0, min(1.0, d)) * PWM_FULL))
        self.bus.sync_write(dx.GOAL_PWM, 2, [(int(i), le16s(v))])

    def cap(self, ids, pwm_units):
        self.bus.set_pwm([int(i) for i in ids], int(pwm_units))

    def torque(self, ids, on):
        self.bus.torque([int(i) for i in ids], on)

    def torque_off_all(self):
        self.bus.torque_off_all()

    def setup_ram(self, ids, kp=KP, cap=CAP_SCRIPT):
        """The runner's RAM settings (P-only position loop, no profile). -> list of ids that did not acknowledge."""
        bad = []
        for i in ids:
            w = self.bus.write
            ok = all([w(i, dx.POS_P, kp, 2), w(i, dx.POS_I, 0, 2), w(i, dx.POS_D, 0, 2), w(i, dx.FF1, 0, 2),
                      w(i, dx.FF2, 0, 2), w(i, dx.PROFILE_ACCEL, 0, 4), w(i, dx.PROFILE_VELOCITY, 0, 4),
                      w(i, dx.GOAL_PWM, int(cap), 2)])
            if not ok:
                bad.append(i)
        return bad

    # ---- EEPROM (Operating Mode only) and reboot
    def set_mode(self, i, mode):
        if mode not in (dx.OP_POSITION, OP_PWM):
            raise Abort("refusing Operating Mode %s" % mode)
        if self.bus.read(i, dx.TORQUE_ENABLE, 1) != 0:
            self.bus.write(i, dx.TORQUE_ENABLE, 0, 1)
            self.sleep(0.01)
            if self.bus.read(i, dx.TORQUE_ENABLE, 1) != 0:
                raise Abort("id %d: torque still on, cannot change the Operating Mode" % i)
        params = struct.pack("<H", dx.OPERATING_MODE) + bytes([mode])
        self.bus._txrx(i, dx.WRITE, params, 0)
        self.sleep(0.02)
        got = self.bus.read(i, dx.OPERATING_MODE, 1)
        if got != mode:
            raise Abort("id %d: Operating Mode reads %s after writing %d" % (i, got, mode))

    def reboot(self, i, wait=3.0):
        self.bus._send(dx.packet(i, REBOOT))
        self.sleep(0.05)
        self.bus.ser.reset_input_buffer()
        t0 = self.now()
        while self.now() - t0 < wait:
            self.sleep(0.2)
            if self.bus.ping(i) is not None:
                self.sleep(0.1)
                return True
        return False


# ====================================================================================================== simulated servos
class SimServos:
    """XL430-like simulation for offline tests. Each joint: J q'' = tau_s(V) (duty - q'/w_nl(V)) - tau_c sign(q')
    + tau_g(q) + tau_stop(q). Position mode duty = clip(P/128 * err_ticks / 885, +-Goal PWM/885). Overload: duty at
    >= 0.5 with |q'| < 0.2 rad/s for 3 s trips (latched, torque off, cleared by reboot). Virtual clock: every bus call
    costs time; sleep() integrates."""

    sim = True

    def __init__(self, contract, q0=None, J=0.006, tau_c=0.02, gravity=0.08, v0=12.2, drop_per_rad=0.0, seed=0):
        self.c = contract
        self.ids = [int(i) for i in contract.ids]
        self.k_of = {i: k for k, i in enumerate(self.ids)}
        n = 18
        self.q = np.array(contract.sleep if q0 is None else q0, dtype=float)
        self.w = np.zeros(n)
        self.goal = self.q.copy()
        self.on = np.zeros(n, bool)
        self.mode = np.full(n, dx.OP_POSITION)
        self.capv = np.full(n, PWM_FULL, float)
        self.dcmd = np.zeros(n)
        self.kp = np.full(n, 640.0)
        self.alarm = np.zeros(n, int)
        self.over = np.zeros(n)
        self.T = np.full(n, 32.0)
        self.duty_out = np.zeros(n)
        self.J, self.tau_c, self.v0 = J, tau_c, v0
        self.g = np.zeros(n)
        self.g[6:18] = gravity  # femur/tibia: tau_g = -g cos(q)
        self.stops = {}  # k -> (q_stop, +1: blocks q > q_stop / -1: blocks q < q_stop)
        self.t = 0.0
        self.last_error = {}
        self.drop_per_rad = drop_per_rad
        self.rng = np.random.RandomState(seed)
        self.prev_goal = self.goal.copy()
        self.n_drop = 0
        self.breaks = {}

    # ---- physics
    def volts(self):
        return self.v0 - 0.6 * float(np.abs(self.duty_out).sum()) / 6.0

    def _duty(self):
        d = np.zeros(18)
        pos = self.on & (self.mode == dx.OP_POSITION)
        err_ticks = (self.goal - self.q) / dx.TICK
        d[pos] = np.clip(self.kp[pos] / 128.0 * err_ticks[pos], -self.capv[pos], self.capv[pos]) / PWM_FULL
        pw = self.on & (self.mode == OP_PWM)
        d[pw] = np.clip(self.dcmd[pw], -1, 1)
        return d

    def _step(self, h):
        V = self.volts()
        ts, wnl = 1.4 * V / 11.1, 5.97 * V / 11.1
        d = self._duty()
        self.duty_out = d
        tau = np.where(self.on, np.clip(ts * (d - self.w / wnl), -ts, ts), 0.0)
        tau = tau - self.tau_c * np.tanh(self.w / 0.05) - self.g * np.cos(self.q) - 0.002 * self.w
        for k, (qs, side) in self.stops.items():
            pen = (self.q[k] - qs) * side
            if pen > 0:
                tau[k] -= side * (80.0 * pen + 0.5 * self.w[k] * side)
        self.w += h * tau / self.J
        self.q += h * self.w
        stall = self.on & (np.abs(d) >= 0.5) & (np.abs(self.w) < 0.2)
        self.over = np.where(stall, self.over + h, np.maximum(self.over - h, 0.0))
        trip = self.over > 3.0
        self.alarm[trip] |= 0x20
        self.on[trip] = False
        self.T += h * (4.0 * d * d - (self.T - 30.0) / 300.0)
        self.t += h

    def advance(self, dt):
        while dt > 1e-9:
            h = min(dt, 0.0005)
            self._step(h)
            dt -= h

    def now(self):
        return self.t

    def sleep(self, dt):
        self.advance(max(0.0, dt))

    def _status(self, k):
        if self.alarm[k]:
            self.last_error[self.ids[k]] = dx.ALERT

    # ---- reads
    def block(self, i):
        k = self.k_of[i]
        t = self.t
        self.advance(0.0012)
        self._status(k)
        return {"t": t, "pwm": round(self.duty_out[k] * PWM_FULL) / PWM_FULL, "load": float(self.duty_out[k]),
                "vel": round(self.w[k] / VEL_UNIT) * VEL_UNIT, "q": dx.pos_to_rad(dx.rad_to_pos(self.q[k])),
                "V": round(self.volts(), 1), "T": float(round(self.T[k])), "alert": bool(self.alarm[k])}

    def _drops(self, ids):
        step = np.abs(self.goal - self.prev_goal)
        self.prev_goal = self.goal.copy()
        p = np.clip(self.drop_per_rad * step, 0, 0.5)
        hit = [i for i in ids if self.rng.rand() < p[self.k_of[i]]]
        if not hit:
            return set(), 0
        cul = hit[0]
        lst = list(ids)
        return set(lst[lst.index(cul):]), cul

    def read_all(self, ids):
        self.advance(0.0065)
        miss, cul = self._drops(ids)
        out = {}
        for i in ids:
            k = self.k_of[i]
            self._status(k)
            if i not in miss:
                out[i] = (self.duty_out[k], self.duty_out[k], self.w[k], dx.pos_to_rad(dx.rad_to_pos(self.q[k])))
        return out, cul

    def read_positions(self, ids):
        self.advance(0.0055)
        miss, cul = self._drops(ids)
        self.last_read_diag = (0, 18 - len(miss), 0, 0, cul)
        out = []
        for i in ids:
            k = self.k_of[i]
            self._status(k)
            out.append(None if i in miss else dx.pos_to_rad(dx.rad_to_pos(self.q[k])))
        return out

    def health(self, i):
        self.advance(0.0008)
        return round(self.volts(), 1), float(round(self.T[self.k_of[i]]))

    def read(self, i, addr, n):
        k = self.k_of[i]
        self.advance(0.001)
        if addr == dx.OPERATING_MODE:
            return int(self.mode[k])
        if addr == dx.TORQUE_ENABLE:
            return int(self.on[k])
        if addr == dx.HW_ERROR:
            return int(self.alarm[k])
        if addr == dx.MODEL_NUMBER:
            return dx.MODEL_XL430
        return 0

    def ping(self, i):
        self.advance(0.001)
        return dx.MODEL_XL430

    def hw_error(self, i):
        return int(self.alarm[self.k_of[i]])

    def take_alerts(self):
        e = dict(self.last_error)
        self.last_error.clear()
        return e

    # ---- writes
    def goals(self, ids, qs):
        self.advance(0.0006)
        for i, q in zip(ids, qs):
            self.goal[self.k_of[int(i)]] = dx.pos_to_rad(dx.rad_to_pos(q))

    def duty(self, i, d):
        self.advance(0.0003)
        self.dcmd[self.k_of[int(i)]] = round(max(-1.0, min(1.0, d)) * PWM_FULL) / PWM_FULL

    def cap(self, ids, pwm_units):
        self.advance(0.0005)
        for i in ids:
            self.capv[self.k_of[int(i)]] = pwm_units

    def torque(self, ids, on):
        self.advance(0.0005)
        for i in ids:
            k = self.k_of[int(i)]
            if on and self.alarm[k]:
                continue  # a latched alarm keeps torque off
            self.on[k] = on

    def torque_off_all(self):
        self.on[:] = False

    def setup_ram(self, ids, kp=KP, cap=CAP_SCRIPT):
        for i in ids:
            k = self.k_of[int(i)]
            self.kp[k], self.capv[k] = kp, cap
        self.advance(0.002 * len(ids))
        return []

    def set_mode(self, i, mode):
        k = self.k_of[int(i)]
        if self.on[k]:
            self.on[k] = False
        self.mode[k] = mode
        self.advance(0.02)

    def reboot(self, i, wait=3.0):
        k = self.k_of[int(i)]
        self.alarm[k] = 0
        self.over[k] = 0.0
        self.on[k] = False
        self.kp[k], self.capv[k] = 640.0, PWM_FULL
        self.advance(0.6)
        return True


# ====================================================================================================== recorder
class Recorder:
    """Rows of named fields -> npz; meta -> json next to it."""

    def __init__(self):
        self.rows = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(v)

    def __len__(self):
        return len(self.rows.get("t", []))

    def arrays(self):
        out = {}
        for k, v in self.rows.items():
            a = np.array(v)
            out[k] = a.astype(np.float64 if k == "t" else np.float32) if a.dtype.kind in "fiub" else a
        return out


def save_run(name, rec, meta, logdir=LOGDIR):
    os.makedirs(logdir, exist_ok=True)
    stem = os.path.join(logdir, "%s_%s" % (time.strftime("%Y-%m-%d_%H-%M-%S"), name))
    k = 1
    while os.path.exists(stem + ".npz"):
        stem = stem.rstrip("0123456789").rstrip("_") + "_%d" % k
        k += 1
    np.savez_compressed(stem + ".npz", **rec.arrays())
    meta = dict(meta, test=meta.get("test", name), saved=time.strftime("%Y-%m-%d %H:%M:%S"), git=git_rev())
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return stem


# ====================================================================================================== bench
class Bench:
    """Shared state + safety for all tests. say(msg) prints; ask(msg) -> bool (y/n prompt, --yes answers yes)."""

    def __init__(self, servos, contract=None, say=print, ask=None, logdir=LOGDIR):
        self.s = servos
        self.c = contract or Contract()
        self.ids = [int(i) for i in self.c.ids]
        self.k_of = {i: k for k, i in enumerate(self.ids)}
        self.say = say
        self.ask = ask or (lambda m: True)
        self.logdir = logdir
        self.pwm_mode = set()  # ids currently in PWM mode (also in PWM_MARK)
        self.saved = []
        self.tolerate = set()  # ids whose (already reported) latched alarm must not abort a recovery move

    def tolerate_alarmed(self):
        """Before a recovery move: stop aborting on the servos that already have a latched alarm."""
        for i in self.ids:
            if self.s.hw_error(i):
                self.tolerate.add(i)
        return sorted(self.tolerate)

    # ---- joints
    def joint(self, name):
        """'RF_coxa', 'right_front_coxa', 'rf coxa' or a servo id -> sim index k."""
        if isinstance(name, int) or str(name).isdigit():
            return self.k_of[int(name)]
        s = str(name).lower().replace(" ", "_").replace("-", "_")
        short = {"rf": "right_front", "rm": "right_middle", "rb": "right_back", "lf": "left_front",
                 "lm": "left_middle", "lb": "left_back"}
        leg, _, typ = s.partition("_")
        if leg in short:
            s = short[leg] + "_" + typ
        if s not in self.c.names:
            raise SystemExit("unknown joint %r (e.g. RF_coxa, LB_femur, or a servo id)" % name)
        return self.c.names.index(s)

    def label(self, k):
        return "%s (id %d)" % (self.c.short(k), self.ids[k])

    # ---- safety
    def check(self, sample=None, extra_ids=()):
        """Raise Abort on a hardware alert, over-temperature or low voltage."""
        al = {i: e for i, e in self.s.take_alerts().items() if i not in self.tolerate}
        if al:
            parts = []
            for i in list(al)[:3]:
                hw = self.s.hw_error(i)
                parts.append("id %d %s" % (i, "?" if hw is None else dx.errors_str(hw)))
            raise Abort("hardware alert: " + "; ".join(parts))
        if sample is not None:
            if sample["T"] >= TEMP_ABORT:
                raise Abort("temperature %.0f C >= %.0f C" % (sample["T"], TEMP_ABORT))
            if 0 < sample["V"] < V_ABORT:
                raise Abort("voltage %.1f V < %.1f V" % (sample["V"], V_ABORT))

    def voltage_survey(self):
        """-> (median V, min V, max T, alarms {id: hw}) over all 18."""
        vs, ts, alarms = [], [], {}
        for i in self.ids:
            h = self.s.health(i)
            if h is not None:
                vs.append(h[0])
                ts.append(h[1])
            hw = self.s.hw_error(i)
            if hw:
                alarms[i] = hw
        if not vs:
            raise Abort("no servo answers the health read (servo power off?)")
        return float(np.median(vs)), float(min(vs)), float(max(ts)), alarms

    # ---- start-up
    def restore_marked(self):
        """Put every servo listed in PWM_MARK (or found in a non-position mode) back to Operating Mode 3."""
        ids = set()
        if os.path.exists(PWM_MARK):
            try:
                ids |= set(json.load(open(PWM_MARK)).get("ids", []))
            except Exception:
                pass
        for i in self.ids:
            m = self.s.read(i, dx.OPERATING_MODE, 1)
            if m is not None and m != dx.OP_POSITION:
                ids.add(i)
        for i in sorted(ids):
            self.s.torque([i], False)
            self.s.set_mode(i, dx.OP_POSITION)
            self.say("restored Operating Mode 3 on id %d (left in PWM mode by an earlier run)" % i)
        self._write_mark(set())

    def _write_mark(self, ids):
        os.makedirs(os.path.dirname(PWM_MARK), exist_ok=True)
        if ids:
            with open(PWM_MARK, "w") as f:
                json.dump({"ids": sorted(ids), "since": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        elif os.path.exists(PWM_MARK):
            os.remove(PWM_MARK)

    def to_pwm_mode(self, i):
        self.s.torque([i], False)
        self.pwm_mode.add(i)
        self._write_mark(self.pwm_mode)  # mark first: a crash between the two lines still gets restored
        self.s.set_mode(i, OP_PWM)
        self.s.duty(i, 0.0)

    def to_position_mode(self, i, kp=KP, cap=CAP_SCRIPT):
        self.s.duty(i, 0.0)
        self.s.torque([i], False)
        self.s.set_mode(i, dx.OP_POSITION)
        self.pwm_mode.discard(i)
        self._write_mark(self.pwm_mode)
        self.s.setup_ram([i], kp, cap)

    def clear_alarms(self, kp=KP, cap=CAP_SCRIPT, ids=None):
        """Reboot every servo with a latched alarm; re-apply the RAM gains. -> list of rebooted ids."""
        done = []
        for i in (ids or self.ids):
            hw = self.s.hw_error(i)
            if hw:
                self.say("id %d: latched hardware error 0x%02x (%s) -> reboot" % (i, hw, dx.errors_str(hw)))
                if not self.s.reboot(i):
                    raise Abort("id %d did not come back after reboot" % i)
                self.s.setup_ram([i], kp, cap)
                hw2 = self.s.hw_error(i)
                if hw2:
                    raise Abort("id %d: error 0x%02x still set after reboot" % (i, hw2))
                done.append(i)
        self.s.take_alerts()
        return done

    def safe_stop(self):
        """Never raises: duty 0 + mode 3 for PWM-mode servos, then torque off everywhere."""
        for i in list(self.pwm_mode):
            try:
                self.to_position_mode(i)
            except Exception as e:
                self.say("WARNING: could not restore Operating Mode 3 on id %d (%s); the next bench start retries" % (i, e))
        try:
            self.s.torque_off_all()
        except Exception:
            pass

    # ---- multi-joint moves (position mode, 50 Hz, like the runner)
    def read_q(self, tries=5):
        for _ in range(tries):
            q = self.s.read_positions(self.ids)
            if all(v is not None for v in q):
                return np.array(q, dtype=float)
        raise Abort("could not read all 18 positions")

    def torque_up(self, cap=CAP_SCRIPT, kp=KP):
        q = self.read_q()
        bad = self.s.setup_ram(self.ids, kp, cap)
        if bad:
            raise Abort("RAM setup not acknowledged by ids %s" % bad)
        self.s.goals(self.ids, q)
        self.s.torque(self.ids, True)
        return q

    def ramp(self, q_from, q_to, duration=None, rate=0.5, rec=None, phase=0, dt=0.02, monitor=None):
        """Cosine-eased joint ramp at 50 Hz. duration default: largest joint distance / rate x pi/2 (>= 1 s)."""
        q_from, q_to = np.asarray(q_from, float), self.c.clip(np.asarray(q_to, float))
        if duration is None:
            duration = max(1.0, float(np.abs(q_to - q_from).max()) / rate * math.pi / 2)
        n = max(1, int(math.ceil(duration / dt)))
        nxt = self.s.now()
        for j in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * j / n)
            q_cmd = q_from + (q_to - q_from) * s
            self.step_all(q_cmd, rec, phase, monitor)
            nxt += dt
            w = nxt - self.s.now()
            if w > 0:
                self.s.sleep(w)
        return q_to

    def hold_all(self, q, seconds, rec=None, phase=0, dt=0.02, monitor=None):
        nxt = self.s.now()
        for _ in range(max(1, int(round(seconds / dt)))):
            self.step_all(q, rec, phase, monitor)
            nxt += dt
            w = nxt - self.s.now()
            if w > 0:
                self.s.sleep(w)

    def step_all(self, q_cmd, rec=None, phase=0, monitor=None):
        """One 50 Hz step: SYNC_WRITE goals, SYNC_READ pwm/load/vel/q of all 18, safety, record."""
        q_cmd = self.c.clip(np.asarray(q_cmd, float))
        self.s.goals(self.ids, q_cmd)
        got, culprit = self.s.read_all(self.ids)
        t = self.s.now()
        nan = float("nan")
        pwm = np.array([got[i][0] if i in got else nan for i in self.ids])
        q = np.array([got[i][3] if i in got else nan for i in self.ids])
        vel = np.array([got[i][2] if i in got else nan for i in self.ids])
        self.check()
        self.n_step = getattr(self, "n_step", 0) + 1
        V = float("nan")
        if self.n_step % 5 == 0:  # battery: one servo's Present Input Voltage every 5th step (~1 ms)
            h = self.s.health(self.ids[(self.n_step // 5) % 18])
            if h is not None:
                V = h[0]
                if V < V_ABORT:
                    raise Abort("voltage %.1f V < %.1f V" % (V, V_ABORT))
        if rec is not None:
            rec.add(t=t, phase=phase, q_cmd=q_cmd, q=q, vel=vel, pwm=pwm, valid=np.isfinite(q), culprit=culprit, V=V)
        if monitor is not None:
            monitor(t, q_cmd, q, pwm)
        return q, pwm


def open_servos(port="/dev/ttyDXL"):
    bus = dx.DxlBus(port)
    return RealServos(bus), bus
