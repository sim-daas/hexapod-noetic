#!/usr/bin/env python3
"""Keyboard teleop + safe policy runner for pxmark4 (blind S1H4 B0). One plain-Python process, no ROS. Python 3.8.

    python3 scripts/hexapod_teleop.py               # on the Pi, in the foreground of the SSH session
    python3 scripts/hexapod_teleop.py --fake        # dry run with a simulated bus + IMU (any machine)

Keys:
  u  stand up (from torque off; belly on the ground)     j  sit down -> sleep -> torque off
  p  policy on/off (on only from STAND; off = freeze, then back to STAND)
  w/s  v_x up/down     a/d  v_y left/right (only when v_x >= 0.05)     q/e  yaw rate left/right
  space  zero command (standing)     f  freeze (hold pose, policy off)     x  torque off now (soft kill)
  c  joint sign check (torque off, lying on the belly)    Esc / Ctrl-C  sit down -> torque off -> exit

Every fault (safety.py) -> freeze 0.5 s -> back to STAND -> sit -> sleep -> torque off. A fall (tilt > 60 deg) or
a dead bus -> torque off at once. SIGHUP (SSH drop) and SIGTERM sit down and exit; a `finally` turns torque off.
Logs: ~/hexapod_logs/<date>_<time>.npz (per step) + .jsonl (events).
"""
import argparse
import json
import os
import select
import signal
import sys
import time
import traceback
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dxl_bus as dx  # noqa: E402
from joint_map import REPO, Contract  # noqa: E402
from poses import Motion, Poses, Segment, joint_ramp  # noqa: E402
from proprio import ProprioReader  # noqa: E402
from safety import FAULT, KILL, LEVEL_NAME, OK, WARN, Safety, tilt_deg  # noqa: E402
from s1h_policy import ObsBuilder  # noqa: E402

OFF, STANDUP, STAND, POLICY, RECOVER, SIT, FROZEN, SIGNCHECK, EXIT = (
    "OFF", "STANDUP", "STAND", "POLICY", "RECOVER", "SIT", "FROZEN", "SIGNCHECK", "EXIT")
STATE_CODE = {s: i for i, s in enumerate((OFF, STANDUP, STAND, POLICY, RECOVER, SIT, FROZEN, SIGNCHECK, EXIT))}
PWM_FULL = 885  # XL430 PWM Limit (100 %); Goal PWM is the torque cap
TORQUE_SCRIPT, TORQUE_POLICY = int(0.70 * PWM_FULL), PWM_FULL  # 70 % scripted moves, 100 % policy (as in the sim)
# Position P gain matching the sim's AX-12A compliance: the AX reaches full duty 32 AX ticks (0.164 rad) past the
# goal; the X-series position loop gives PWM = (P / 128) * error[XL ticks], full (885) at 0.164 rad = 106.8 XL ticks
# -> P = 885 * 128 / 106.8 = 1061. I, D, feed-forward 0 and no motion profile: the sim models P-only control.
KP_AX_MATCH = 1061
# "Lying down, legs folded" check before u / c, per joint type (coxa, femur, tibia), rad from the vendor sleep pose.
# With torque off on the belly the legs relax: measured 2026-11-06 femur up to 0.4 rad and tibia up to 0.85 rad off
# (the feet rest on the ground and open the knee). The stand-up / sign check first ramp to the exact SLEEP pose.
SLEEP_TOL_TYPE = (0.3, 0.6, 1.0)
SLEEP_TOL = np.repeat(SLEEP_TOL_TYPE, 6)
CONFIRM_S = 3.0
CMD_GATE_S = 0.2
SIGN_PULSE = 0.15
ESC = "\x1b"


def hold(q, seconds, name="hold"):
    q = np.array(q, dtype=float)
    return Segment(name, seconds, lambda s: q.copy(), stretch=False)


# ---------------------------------------------------------------------------------------------------- teleop command
class Command:
    """Latched twist command, snapped to the trained set:
    standing (all 0) | turn on the spot (v_x = v_y = 0, 0.1 <= |w| <= 0.4) | walking (v_x in [0.05, 0.2],
    |v_y| <= 0.05, |w| <= 0.4). Key changes reach the policy at most once per 0.2 s."""

    def __init__(self, contract, vx_max=0.10, wz_max=0.2):
        c = contract.command
        self.vx_lo, self.vx_hi = c["lin_vel_x"]
        self.vy_hi = c["lin_vel_y"][1]
        self.wz_hi = c["ang_vel_z"][1]
        self.turn_lo = c["turn_yaw_range"][0]
        self.vx_max = min(vx_max, self.vx_hi)
        self.wz_max = min(wz_max, self.wz_hi)
        self.vx_steps = [0.0] + [round(v, 4) for v in np.arange(self.vx_lo, self.vx_max + 1e-9, 0.025)]
        self.vx_i, self.vy, self.wz = 0, 0.0, 0.0
        self.applied = np.zeros(3)
        self.t_applied = -1e9

    @property
    def target(self):
        return self.snap(self.vx_steps[self.vx_i], self.vy, self.wz)

    def snap(self, vx, vy, wz):
        vx = float(np.clip(vx, 0.0, self.vx_max))
        if vx < self.vx_lo:  # standing or turn on the spot
            vx, vy = 0.0, 0.0
            if abs(wz) > 1e-6:
                wz = float(np.sign(wz) * np.clip(abs(wz), self.turn_lo, self.wz_max))
            else:
                wz = 0.0
        else:
            vy = float(np.clip(vy, -self.vy_hi, self.vy_hi))
            wz = float(np.clip(wz, -self.wz_max, self.wz_max))
        return np.array([vx, vy, wz])

    def key(self, k):
        if k == "w":
            self.vx_i = min(self.vx_i + 1, len(self.vx_steps) - 1)
        elif k == "s":
            self.vx_i = max(self.vx_i - 1, 0)
        elif k in ("a", "d") and self.vx_steps[self.vx_i] >= self.vx_lo:
            self.vy = float(np.clip(self.vy + (0.025 if k == "a" else -0.025), -self.vy_hi, self.vy_hi))
        elif k in ("q", "e"):
            self.wz = float(np.clip(self.wz + (0.1 if k == "q" else -0.1), -self.wz_max, self.wz_max))
            if abs(self.wz) < 1e-6:
                self.wz = 0.0
        elif k == " ":
            self.zero()
            self.t_applied = -1e9  # stop applies immediately
        else:
            return False
        if self.vx_steps[self.vx_i] < self.vx_lo:
            self.vy = 0.0
        return True

    def zero(self):
        self.vx_i, self.vy, self.wz = 0, 0.0, 0.0

    def update(self, t):
        tgt = self.target
        if not np.allclose(tgt, self.applied) and t - self.t_applied >= CMD_GATE_S:
            self.applied = tgt
            self.t_applied = t
        return self.applied


# ---------------------------------------------------------------------------------------------------- keyboard
class Keys:
    """Non-blocking single keys from a tty in cbreak mode (Ctrl-C still raises SIGINT). EOF -> 'HUP'."""

    def __init__(self, stream=sys.stdin):
        self.fd = stream.fileno() if stream is not None and stream.isatty() else None
        self.old = None
        if self.fd is not None:
            import termios
            import tty
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def poll(self):
        if self.fd is None:
            return []
        out = []
        while select.select([self.fd], [], [], 0)[0]:
            try:
                b = os.read(self.fd, 64)
            except OSError:
                return out + ["HUP"]
            if not b:
                return out + ["HUP"]
            s = b.decode("utf-8", "ignore")
            i = 0
            while i < len(s):
                if s[i] == ESC and i + 1 < len(s) and s[i + 1] in "[O":  # arrow / function key: skip the sequence
                    i += 2
                    while i < len(s) and not s[i].isalpha() and s[i] != "~":
                        i += 1
                    i += 1
                    continue
                out.append(s[i])
                i += 1
        return out

    def restore(self):
        if self.old is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass


# ---------------------------------------------------------------------------------------------------- logging
class Log:
    def __init__(self, directory, enabled=True):
        self.enabled = enabled
        self.rows = {}
        self.path = None
        if enabled:
            os.makedirs(directory, exist_ok=True)
            stem = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.path = os.path.join(directory, stem)
            self.ev = open(self.path + ".jsonl", "a", buffering=1)
        self.t0 = time.monotonic()

    def event(self, kind, **kw):
        kw.update(t=round(time.monotonic(), 4), kind=kind)
        if self.enabled:
            try:
                self.ev.write(json.dumps(kw, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            except Exception:
                pass

    def row(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(np.asarray(v, dtype=np.float32 if k != "t" else np.float64))

    def save(self):
        if not self.enabled or not self.rows:
            return None
        np.savez_compressed(self.path + ".npz", **{k: np.array(v) for k, v in self.rows.items()})
        return self.path + ".npz"


# ---------------------------------------------------------------------------------------------------- fake IMU
class FakeImu:
    """Level, still robot through the inverse of the calibration (for --fake and the offline tests)."""

    def __init__(self, blk, tilt=None):
        self.blk, self.tilt = blk, tilt  # tilt(): optional body-frame gravity unit vector override
        self.fail = False
        self.rng = np.random.RandomState(0)

    def read(self):
        if self.fail:
            raise OSError("fake IMU failure")
        up = np.array([0.0, 0.0, 1.0]) if self.tilt is None else -np.asarray(self.tilt())
        a_body = 9.80665 * up / np.linalg.norm(up)
        a_raw = self.blk.R.T @ a_body / self.blk.sc + self.blk.off
        return a_raw + self.rng.randn(3) * 0.01, self.blk.gyro_bias + self.rng.randn(3) * 0.002


# ---------------------------------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, contract, bus, imu, imu_block, policy, vx_max=0.10, wz_max=0.2, clock=time.monotonic,
                 out=sys.stdout, log=None):
        self.c, self.bus, self.imu, self.blk, self.policy = contract, bus, imu, imu_block, policy
        self.clock, self.out = clock, out
        self.log = log or Log("", enabled=False)
        self.ids = [int(i) for i in contract.ids]
        self.poses = Poses(contract)
        self.reader = ProprioReader(18, contract.dt)
        self.reader_init = False
        self.safety = Safety(18, contract.dt)
        self.cmd = Command(contract, vx_max, wz_max)
        self.obs = ObsBuilder(contract)
        self.state = OFF
        self.torque_on = False
        self.motion = None
        self.motion_prev_q = None
        self.after_motion = None  # state to enter when the motion ends
        self.q_hold = None
        self.q_cmd_last = None
        self.frozen_from = None
        self.core_k = None  # stand-up core segment index while standing up (for the reverse path)
        self.exit_req = None
        self.pending_exit = None  # set by signal handlers, handled at the start of the next tick
        self.n_signals = 0
        self.exit_after_sit = False
        self.qcmd_prev_pol_check = None
        self.policy_start = False
        self.a_prev = np.zeros(18)
        self.qcmd_prev_pol = contract.q_default.copy()
        self.confirm_until = -1.0
        self.sign = None
        self.x9 = np.array([0, 0, 9.80665, 0, 0, 0, 0, 0, -1.0])
        self.hr = 0
        self.hr_tick = 0
        self.t_prev = None
        self.compute = deque(maxlen=250)
        self.periods = deque(maxlen=250)
        self.all_compute, self.all_period = [], []
        self.n_reads = self.n_drops = 0
        self.last_warn = ""
        self.t_status = -1.0
        self.out_dead = False
        self.last_level = (OK, "")
        self.fault_log = []

    # ------------------------------------------------------------------ output
    def say(self, msg):
        self.log.event("msg", text=msg)
        if self.out_dead or self.out is None:
            return
        try:
            self.out.write("\r\x1b[K" + msg + "\n")
            self.out.flush()
        except Exception:
            self.out_dead = True

    def status(self, t):
        if self.out_dead or self.out is None or t - self.t_status < 0.2:
            return
        self.t_status = t
        cp = np.percentile(self.compute, 99) * 1e3 if self.compute else 0.0
        dr = self.n_drops / max(1, self.n_reads)
        v = self.cmd.applied
        tl = tilt_deg(self.x9[6:9])
        line = "%-9s cmd=(%.3f %+.3f %+.2f) p99=%4.1fms drop=%4.1f%% V=%4.1f T=%2.0fC tilt=%2.0f %s" % (
            self.state, v[0], v[1], v[2], cp, 100 * dr, self.safety.v_batt, self.safety.max_temp, tl,
            self.last_warn[:40])
        try:
            self.out.write("\r\x1b[K" + line)
            self.out.flush()
        except Exception:
            self.out_dead = True

    # ------------------------------------------------------------------ bus helpers
    def set_torque_limit(self, value):
        self.bus.set_pwm(self.ids, value)

    def torque_up(self, q_now):
        """Goal = present pose, 70 % torque limit, torque on."""
        self.bus.write_goals(self.ids, q_now)
        self.set_torque_limit(TORQUE_SCRIPT)
        self.bus.torque(self.ids, True)
        self.torque_on = True
        self.q_cmd_last = np.array(q_now, dtype=float)
        self.log.event("torque", on=True)

    def torque_down(self, why):
        self.bus.torque(self.ids, False)
        self.bus.torque_off_all()
        self.torque_on = False
        self.motion = None
        self.state = OFF
        self.q_cmd_last = None
        self.log.event("torque", on=False, why=why)
        self.say("torque OFF (%s)" % why)

    # ------------------------------------------------------------------ transitions
    def run_motion(self, segments, state, after):
        self.motion = Motion(segments, self.c.dt)
        self.motion_prev_q = None
        self.state = state
        self.after_motion = after
        self.log.event("motion", state=state, segments=[(s.name, round(s.duration, 2)) for s in segments])

    def start_standup(self):
        if not self.reader_init or self.reader.mask.min() < 1:
            return self.say("stand-up refused: not all 18 servos read this step")
        lvl, why = self.last_level
        if lvl >= FAULT:
            return self.say("stand-up refused: %s" % why)
        q = self.reader.q.copy()
        d = np.abs(q - self.poses.SLEEP)
        slow = False
        if (d > SLEEP_TOL).any():
            d = d - SLEEP_TOL
            if self.clock() > self.confirm_until:
                self.confirm_until = self.clock() + CONFIRM_S
                k = int(d.argmax())
                return self.say("not lying folded: %s is %.2f rad beyond the tolerance from SLEEP. Press u again "
                                "within %.0f s for a slow ramp to SLEEP" % (self.c.short(k), d[k], CONFIRM_S))
            slow = True
        self.confirm_until = -1.0
        self.torque_up(q)
        self.run_motion(self.poses.standup(q, slow), STANDUP, STAND)
        self.say("standing up (%.1f s)" % sum(s.duration for s in self.motion.segs))

    def start_sit(self, freeze, why, exit_after=False):
        self.exit_after_sit = self.exit_after_sit or exit_after
        if self.state == EXIT:
            return
        if not self.torque_on or self.state == OFF:
            if self.exit_after_sit:
                self.state = EXIT
            return
        if self.state == SIT:
            return
        q = self.reader.q.copy()
        segs = [hold(q, 0.5, "freeze")] if freeze else []
        src = self.frozen_from if self.state == FROZEN else self.state
        if self.torque_on:
            self.set_torque_limit(TORQUE_SCRIPT)
        if src == SIGNCHECK:
            segs += [self.poses.to_pose("to_sleep", q, self.poses.SLEEP, 1.0)]
        elif src == STANDUP and self.core_k is not None:
            segs += self.poses.sitdown_from_core(self.core_k, q)
        elif src == STANDUP:  # still in the initial ramp to SLEEP
            segs += [self.poses.to_pose("to_sleep", q, self.poses.SLEEP, 1.0)]
        else:
            segs += [self.poses.to_pose("to_stand", q, self.poses.STAND, 2.0)] + self.poses.sitdown_from_stand()
        segs += [hold(self.poses.SLEEP, 0.5, "sleep")]
        self.run_motion(segs, SIT, OFF)
        self.say("sitting down (%s), %.1f s" % (why, sum(s.duration for s in segs)))

    def start_policy(self):
        if self.policy is None:
            return self.say("no policy loaded")
        self.set_torque_limit(TORQUE_POLICY)
        self.cmd.zero()
        self.cmd.applied = np.zeros(3)
        self.policy.reset()
        self.policy_start = True
        self.a_prev = np.zeros(18)
        self.qcmd_prev_pol = self.c.q_default.copy()
        self.state = POLICY
        self.say("POLICY ON  cmd=%s" % np.round(self.cmd.applied, 3))

    def stop_policy(self, why="policy off"):
        self.set_torque_limit(TORQUE_SCRIPT)
        q = self.reader.q.copy()
        self.run_motion([hold(q, 0.5, "freeze"), self.poses.to_pose("to_stand", q, self.poses.STAND, 2.0)],
                        RECOVER, STAND)
        self.say("%s: freeze, back to STAND" % why)

    def freeze(self):
        if not self.torque_on or self.state not in (STANDUP, STAND, POLICY, RECOVER):
            return self.say("f: freeze only while standing up, standing, walking or recovering (x = torque off)")
        if self.state == POLICY:
            self.set_torque_limit(TORQUE_SCRIPT)
        self.frozen_from = self.state
        self.q_hold = self.reader.q.copy()
        self.motion = None
        self.state = FROZEN
        self.say("FROZEN (from %s): j = sit, x = torque off" % self.frozen_from)

    def start_signcheck(self):
        if self.state != OFF:
            return self.say("sign check only from torque off, lying on the belly")
        if not self.reader_init or self.reader.mask.min() < 1:
            return self.say("sign check refused: not all servos read")
        q = self.reader.q.copy()
        d = np.abs(q - self.poses.SLEEP) - SLEEP_TOL
        if (d > 0).any():
            k = int(d.argmax())
            return self.say("sign check refused: %s is %.2f rad beyond the tolerance from SLEEP (lie the robot "
                            "down, legs folded)" % (self.c.short(k), d[k]))
        self.torque_up(q)
        self.sign = {"i": -1, "answers": {}, "await": False}
        self.run_motion([self.poses.to_pose("to_sleep", q, self.poses.SLEEP, 1.5)], SIGNCHECK, SIGNCHECK)
        self.say("sign check: each joint pulses %.2f rad and back; answer y (moved as described) / n / r (repeat)"
                 % SIGN_PULSE)

    def _sign_next(self):
        s = self.sign
        s["i"] += 1
        if s["i"] >= 18:
            bad = [k for k, v in s["answers"].items() if v != "y"]
            self.log.event("sign_check", answers=s["answers"], failed=bad)
            self.say("sign check done: %s" % ("ALL 18 OK" if not bad else "FAILED: %s" % ", ".join(bad)))
            self.run_motion([hold(self.poses.SLEEP, 0.3)], SIT, OFF)
            return
        self._sign_pulse()

    def _sign_pulse(self):
        i = self.sign["i"]
        t = self.c.names[i].rsplit("_", 1)[1]
        side = "right" if self.c.names[i].startswith("right") else "left"
        d = {"coxa": (+SIGN_PULSE, "foot sweeps counter-clockwise seen from above -> %s leg moves %s"
                      % (side, "FORWARD" if side == "right" else "BACKWARD")),
             "femur": (-SIGN_PULSE, "whole leg swings UP about the hip (knee toward the body), foot lifts"),
             "tibia": (-SIGN_PULSE, "lower leg opens: foot tip moves UP and OUTWARD")}[t]
        q0 = self.poses.SLEEP
        q1 = q0.copy()
        q1[i] += d[0]
        q1 = self.c.clip(q1)
        self.say("[%2d/18] %-12s %+.2f rad: expect %s" % (i + 1, self.c.short(i), d[0], d[1]))
        self.run_motion([joint_ramp("pulse", q0, q1, 0.5), hold(q1, 0.4), joint_ramp("back", q1, q0, 0.5)],
                        SIGNCHECK, SIGNCHECK)
        self.sign["await"] = False

    # ------------------------------------------------------------------ keys
    def on_key(self, k):
        self.log.event("key", key=k, state=self.state)
        if k == "HUP":
            return self.request_exit("stdin closed")
        if k == ESC:
            return self.request_exit("Esc")
        if k == "x":
            if self.torque_on:
                self.torque_down("x key")
            return
        if self.state == SIGNCHECK and self.sign is not None and self.sign["await"] and k in "ynr":
            if k == "r":
                return self._sign_pulse()
            self.sign["answers"][self.c.names[self.sign["i"]]] = k
            return self._sign_next()
        if k == "f":
            return self.freeze()
        if k == "j":
            return self.start_sit(freeze=self.state == POLICY, why="j key")
        if k == "u":
            if self.state == OFF:
                return self.start_standup()
            return self.say("u: only from torque off")
        if k == "c":
            return self.start_signcheck()
        if k == "p":
            if self.state == STAND:
                return self.start_policy()
            if self.state == POLICY:
                return self.stop_policy()
            return self.say("p: policy only from STAND")
        if k in "wsadqe ":
            if self.state != POLICY:
                return self.say("%r: commands only while the policy runs (p from STAND)" % k)
            self.cmd.key(k)
            self.log.event("cmd_target", target=self.cmd.target)

    def request_exit(self, why):
        if self.exit_req is None:
            self.exit_req = why
            self.log.event("exit_request", why=why)
            self.say("exit requested (%s): sit down, torque off, exit" % why)
        self.start_sit(freeze=self.state == POLICY, why=why, exit_after=True)

    # ------------------------------------------------------------------ fault reaction
    def react(self, level, why):
        if level == KILL:
            self.say("KILL: %s" % why)
            self.fault_log.append(why)
            if self.torque_on:
                self.torque_down(why)
            if why.startswith("bus dead"):
                self.state = EXIT
            return
        if level == FAULT and self.torque_on and self.state not in (SIT, OFF):
            self.say("FAULT: %s" % why)
            self.fault_log.append(why)
            self.log.event("fault", why=why, state=self.state)
            self.start_sit(freeze=True, why="fault: " + why)

    # ------------------------------------------------------------------ one control step
    def tick(self, keys=()):
        t = self.clock()
        period = self.c.dt if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        if self.pending_exit is not None:
            why, self.pending_exit = self.pending_exit, None
            self.request_exit(why)
        if self.exit_req is not None and self.state == OFF:
            self.state = EXIT
        for k in keys:
            self.on_key(k)

        # ---- reads
        raw = self.bus.read_positions(self.ids)
        valid = np.array([r is not None for r in raw])
        q_raw = np.array([r if r is not None else 0.0 for r in raw])
        self.n_reads += 18
        self.n_drops += int((~valid).sum())
        if not self.reader_init:
            if valid.all():
                self.reader.reset(q_raw)
                self.reader_init = True
        elif self.policy_start:
            self.reader.reset(np.where(valid, q_raw, self.reader.q))  # episode start, as hexasim's env reset
        if self.reader_init:
            self.reader.update(q_raw, valid)
        try:
            a_raw, w_raw = self.imu.read()
            self.x9 = self.blk.update(a_raw, w_raw, t)
            self.safety.imu_ok(t)
        except OSError as e:
            self.log.event("imu_error", err=str(e))
        alarm = None
        if getattr(self.bus, "last_error", None):  # status packets with the hardware-alert bit
            ids = list(self.bus.last_error)
            self.bus.last_error.clear()
            parts = []
            for k in ids[:2]:
                hw = self.bus.hw_error(k)
                if hw is None or hw & dx.ALARM_MASK:
                    parts.append("id %d %s" % (k, "?" if hw is None else dx.errors_str(hw)))
            alarm = "; ".join(parts) or None

        # ---- monitors (on the command the servos were tracking until now)
        lvl, why = self.safety.step(t, valid, self.reader.q, self.q_cmd_last, self.x9[6:9], period, self.torque_on,
                                    alarm)
        self.last_level = (lvl, why)
        self.last_warn = why if lvl >= WARN else ""
        if lvl >= FAULT:
            self.react(lvl, why)

        # ---- command
        q_cmd, a, o_p = self.command(t)
        if self.state == POLICY and a is not None:
            pl, pw = Safety.check_policy(q_cmd, self.qcmd_prev_pol_check, a)
            if pl >= FAULT:
                self.react(pl, pw)
                q_cmd, a, o_p = self.command(t)
        if q_cmd is not None and self.torque_on:
            q_cmd = self.c.clip(q_cmd)
            self.bus.write_goals(self.ids, q_cmd)
            self.q_cmd_last = q_cmd
        self.policy_start = False
        # ---- health (after the command went out; every 2nd step: one servo per read, a full round in 0.72 s)
        if self.hr_tick % 2 == 0:
            sid = self.ids[self.hr % 18]
            self.hr += 1
            h = self.bus.health(sid)
            if h is not None:
                self.safety.health(sid, *h)
        self.hr_tick += 1
        dt_c = self.clock() - t
        self.compute.append(dt_c)
        self.periods.append(period)
        self.all_compute.append(dt_c)
        self.all_period.append(period)
        nan18 = np.full(18, np.nan)
        self.log.row(t=t, state=STATE_CODE[self.state], q=self.reader.q, valid=valid,
                     q_cmd=q_cmd if q_cmd is not None else nan18, a=a if a is not None else nan18,
                     o_p=o_p if o_p is not None else np.full(111, np.nan), cmd=self.cmd.applied, imu9=self.x9,
                     period=period, compute=dt_c, level=lvl)
        self.status(t)
        return q_cmd

    def command(self, t):
        """-> (q_cmd or None, a or None, o_p or None) for the current state."""
        cmd = self.cmd.update(t) if self.state == POLICY else np.zeros(3)
        if not self.torque_on or self.state in (OFF, EXIT):
            return None, None, None
        if self.motion is not None and self.state in (STANDUP, SIT, RECOVER, SIGNCHECK):
            q = self.motion.next()
            if self.state == STANDUP:  # core segment of the returned q (segment 0 = ramp to SLEEP)
                self.core_k = self.motion.i - 1 if self.motion.i >= 1 else None
            if q is not None:
                if self.motion_prev_q is not None:
                    self.safety.check_script(q, self.motion_prev_q)
                self.motion_prev_q = q
                return q, None, None
            return self.motion_finished()
        if self.state == STAND:
            return self.poses.STAND.copy(), None, None
        if self.state in (FROZEN, SIGNCHECK):
            return self.q_hold.copy(), None, None
        if self.state == POLICY:
            o_p = self.obs.frame(self.reader.q, self.reader.qd, self.reader.mask, self.qcmd_prev_pol, self.a_prev,
                                 self.x9)
            a = self.policy.act(o_p, cmd)
            q_cmd = self.policy.q_cmd(a)
            self.qcmd_prev_pol_check = self.qcmd_prev_pol
            self.a_prev, self.qcmd_prev_pol = a, q_cmd
            return q_cmd, a, o_p
        return None, None, None

    def motion_finished(self):
        after, last = self.after_motion, self.motion_prev_q
        self.motion = None
        if self.state == STANDUP:
            self.core_k = None
            self.say("STAND reached: p = policy, j = sit")
        if self.state == SIGNCHECK and self.sign is not None:
            if self.sign["i"] < 0:
                self._sign_next()
                return self.command(self.clock())
            self.sign["await"] = True
            self.q_hold = self.poses.SLEEP.copy()
            return self.poses.SLEEP.copy(), None, None
        if after == OFF:
            self.torque_down("sit-down complete")
            if self.exit_after_sit:
                self.state = EXIT
            return None, None, None
        self.state = after
        if after == STAND:
            return self.poses.STAND.copy(), None, None
        return last, None, None

    # ------------------------------------------------------------------ real-time loop
    def run(self, keys):
        dt = self.c.dt
        nxt = self.clock()
        errors = 0
        while self.state != EXIT:
            try:
                self.tick(keys.poll())
                errors = 0
            except Exception as e:  # keep the loop alive: sit down without the failing part, or give up
                errors += 1
                self.log.event("exception", err=repr(e), tb=traceback.format_exc())
                if errors == 1:
                    self.say("EXCEPTION: %r" % e)
                if self.state == POLICY or (self.torque_on and self.state not in (SIT,)):
                    self.policy_start = False
                    try:
                        self.start_sit(freeze=True, why="exception")
                    except Exception:
                        pass
                if errors >= 25:
                    self.say("25 consecutive failed steps: torque off, exit")
                    self.bus.torque_off_all()
                    break
            nxt += dt
            wait = nxt - self.clock()
            if wait > 0:
                time.sleep(wait)
            elif wait < -dt:
                nxt = self.clock()

    def summary(self):
        c, p = np.array(self.all_compute) * 1e3, np.array(self.all_period) * 1e3
        if not len(c):
            return "no steps"
        return ("steps %d | compute p50 %.1f p99 %.1f max %.1f ms | period p50 %.1f p99 %.1f max %.1f ms | "
                "dropped reads %.2f %% | faults: %s" % (len(c), np.percentile(c, 50), np.percentile(c, 99), c.max(),
                                                      np.percentile(p, 50), np.percentile(p, 99), p.max(),
                                                      100.0 * self.n_drops / max(1, self.n_reads),
                                                      "; ".join(self.fault_log) or "none"))


# ---------------------------------------------------------------------------------------------------- start-up
def setup_servos(bus, contract, say, kp=KP_AX_MATCH):
    """Ping (XL430), Return Delay 0, position mode, position limits cover the clip box, PWM limit; then RAM settings
    (P-only position loop, no profile, 70 % Goal PWM). Never writes EEPROM. Raises SystemExit."""
    ids = [int(i) for i in contract.ids]
    models = {i: bus.ping(i) for i in ids}
    missing = [contract.names[k] for k, i in enumerate(ids) if models[i] is None]
    if missing:
        raise SystemExit("servos not answering: %s" % ", ".join(missing))
    wrong = [i for i in ids if models[i] != dx.MODEL_XL430]
    if wrong:
        raise SystemExit("unexpected servo model on ids %s: %s (expected XL430-W250 = %d)"
                         % (wrong, [models[i] for i in wrong], dx.MODEL_XL430))
    for k, i in enumerate(ids):
        n = contract.names[k]
        rd, om = bus.read(i, dx.RETURN_DELAY, 1), bus.read(i, dx.OPERATING_MODE, 1)
        lo_p, hi_p = bus.read(i, dx.MIN_POSITION, 4), bus.read(i, dx.MAX_POSITION, 4)
        pwm = bus.read(i, dx.PWM_LIMIT, 2)
        if rd != 0:
            raise SystemExit("%s (id %d): Return Delay Time = %s, need 0" % (n, i, rd))
        if om != dx.OP_POSITION:
            raise SystemExit("%s (id %d): Operating Mode = %s, need 3 (position)" % (n, i, om))
        lo, hi = dx.rad_to_pos(contract.lo[k]), dx.rad_to_pos(contract.hi[k])
        if lo_p is None or hi_p is None or lo_p > lo or hi_p < hi:
            raise SystemExit("%s (id %d): EEPROM position limits [%s, %s] do not cover the clip range [%d, %d]"
                             % (n, i, lo_p, hi_p, lo, hi))
        if pwm is None or pwm < PWM_FULL:
            raise SystemExit("%s (id %d): EEPROM PWM Limit %s < %d" % (n, i, pwm, PWM_FULL))
    for i in ids:
        ok = all([bus.write(i, dx.POS_P, kp, 2), bus.write(i, dx.POS_I, 0, 2), bus.write(i, dx.POS_D, 0, 2),
                  bus.write(i, dx.FF1, 0, 2), bus.write(i, dx.FF2, 0, 2),
                  bus.write(i, dx.PROFILE_ACCEL, 0, 4), bus.write(i, dx.PROFILE_VELOCITY, 0, 4),
                  bus.write(i, dx.GOAL_PWM, TORQUE_SCRIPT, 2)])
        if not ok:
            raise SystemExit("id %d: RAM setup write not acknowledged" % i)
    say("servos: 18 x XL430 ping, Return Delay 0, position mode, limits OK; P %d I 0 D 0, no profile, Goal PWM %d"
        % (kp, TORQUE_SCRIPT))


def start_imu(imu, blk, dt, say, tries=3):
    for k in range(tries):
        samples = []
        for _ in range(100):
            samples.append(imu.read())
            time.sleep(dt)
        try:
            blk.start(samples)
            say("IMU: gyro bias re-measured, gravity seeded")
            return
        except ValueError as e:
            say("IMU start-up (%d/%d): %s - keep the robot still" % (k + 1, tries, e))
    raise SystemExit("IMU start-up failed")


GOV = "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor"


def set_governor(value):
    """Set all 4 cores' cpufreq governor (passwordless sudo on the Pi). -> previous value, or None if not possible.
    performance: in-loop inference p99 5.5 ms vs 9.8 ms under ondemand (1 thread), which clocks down while the loop
    sleeps between steps."""
    import subprocess
    try:
        prev = open(GOV % 0).read().strip()
    except (IOError, OSError):
        return None
    if prev == value:
        return prev
    paths = [GOV % i for i in range(4) if os.path.exists(GOV % i)]
    r = subprocess.run(["sudo", "-n", "tee"] + paths, input=value.encode(), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    return prev if r.returncode == 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyDXL")
    ap.add_argument("--sync-timeout", type=float, default=0.010, help="s for all 18 replies of one SYNC_READ")
    ap.add_argument("--kp", type=int, default=KP_AX_MATCH, help="XL430 position P gain (AX-12A match: %d)"
                    % KP_AX_MATCH)
    ap.add_argument("--vx-max", type=float, default=0.10)
    ap.add_argument("--wz-max", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime threads (2: inference p99 5.5 ms on the Pi)")
    ap.add_argument("--keep-governor", action="store_true", help="do not switch the CPU governor to performance")
    ap.add_argument("--fake", action="store_true", help="simulated bus + IMU (dry run)")
    ap.add_argument("--no-policy", action="store_true", help="poses only (no onnxruntime needed)")
    ap.add_argument("--logdir", default=os.path.expanduser("~/hexapod_logs"))
    args = ap.parse_args(argv)
    if args.vx_max > 0.2 + 1e-9 or args.wz_max > 0.4 + 1e-9:
        raise SystemExit("caps above the trained range (v_x 0.20, w_z 0.4)")

    sys.path.insert(0, HERE)
    from imu_block import ImuBlock, MPU6050

    contract = Contract()
    log = Log(args.logdir)
    blk = ImuBlock(os.path.join(REPO, "config", "imu_calib.json"))
    out = sys.stdout

    def say(m):
        out.write(m + "\n")
        out.flush()
        log.event("msg", text=m)

    if args.fake:
        bus = dx.FakeBus({int(i): float(q) for i, q in zip(contract.ids, contract.sleep)}, dt=contract.dt)
        imu = FakeImu(blk)
    else:
        bus = dx.DxlBus(args.port, sync_timeout=args.sync_timeout)
        imu = MPU6050()
    policy = None
    keys = None
    runner = None
    gov_prev = None
    if not args.fake and not args.keep_governor:
        gov_prev = set_governor("performance")
        say("CPU governor: performance for this session (was %s)" % gov_prev if gov_prev
            else "WARNING: could not set the CPU governor to performance (loop timing may overrun)")
    try:
        setup_servos(bus, contract, say, args.kp)
        say("SLEEP pose: %s" % contract.sleep_source)
        start_imu(imu, blk, contract.dt, say)
        if not args.no_policy:
            from s1h_policy import S1HPolicy
            policy = S1HPolicy(contract, threads=args.threads)
            say("policy: models/control.onnx (%s, obs %d, W %d)" % (contract.raw["source"]["task_id"],
                                                                    contract.obs_dim, contract.window))
        runner = Runner(contract, bus, imu, blk, policy, args.vx_max, args.wz_max, out=out, log=log)
        if args.fake:
            orig_tick = runner.tick

            def tick(keys=()):
                bus.advance()
                return orig_tick(keys)
            runner.tick = tick
        def on_signal(signum, frame, names={signal.SIGHUP: "SIGHUP", signal.SIGTERM: "SIGTERM",
                                            signal.SIGINT: "Ctrl-C"}):
            runner.n_signals += 1
            if runner.n_signals >= 3:  # third signal: stop now (finally turns torque off)
                raise SystemExit("3 signals: immediate exit")
            runner.pending_exit = names.get(signum, str(signum))

        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_signal)
        keys = Keys()
        say(__doc__.split("Keys:")[1].split("Every fault")[0].rstrip())
        say("log: %s.{npz,jsonl}   ready, torque OFF" % log.path)
        runner.run(keys)
    finally:
        try:
            bus.torque_off_all()
        finally:
            if keys is not None:
                keys.restore()
            if gov_prev and gov_prev != "performance":
                set_governor(gov_prev)
            path = log.save()
            if runner is not None:
                say("\n" + runner.summary())
            if path:
                say("saved %s" % path)
            bus.close()


if __name__ == "__main__":
    main()
