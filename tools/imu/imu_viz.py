#!/usr/bin/env python3
"""Live 3-D check of the calibrated IMU on this workstation: the real robot's tilt/yaw drives the URDF base mesh.

  python3 tools/imu/imu_viz.py --live            # streams from the Pi over the reverse tunnel (ssh -p 2222)
  python3 tools/imu/imu_viz.py --replay FILE     # a saved session (tools/imu/logs/*.jsonl)
  python3 tools/imu/imu_viz.py --demo            # synthetic motion, no robot
  add --save-png out.png to run headless (no window): process everything, save the last frame, print the report

What to do (--live): keep the robot still for the first 2 s (gyro bias), then pick the robot up and, in any order,
tilt the nose up/down, lift the left/right side, and spin it about the vertical both ways. The picture must move the way
your hands move the real robot: front end up <-> mesh front end up, left side up <-> mesh left side up, counter-
clockwise seen from above <-> mesh turns counter-clockwise. The checklist on the right fills in as each motion is done
and the gyro<->accel consistency check (see imu_check.py) runs on everything recorded. Keys: q quit, r reset.
The session is saved to tools/imu/logs/ and the final report is printed when you close the window.
"""
import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "scripts")]
import imu_check  # noqa: E402
import imu_synth  # noqa: E402
from imu_block import ImuBlock  # noqa: E402
from urdf_geom import legs as load_legs, load_stl  # noqa: E402

CALIB = os.path.join(HERE, "..", "..", "config", "imu_calib.json")


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Attitude:
    """World <- body rotation for display. Tilt comes from the block's gravity estimate; heading is integrated from
    the gyro (it drifts slowly, there is no compass). World: z up, x = the robot's heading when the session started."""

    def __init__(self):
        self.R = np.eye(3)

    def step(self, g, w, dt):
        u = -np.asarray(g, float)
        u /= np.linalg.norm(u)
        Rp = self.R @ imu_synth.rot_exp(np.asarray(w, float) * dt)  # body-frame rates: right-multiply
        psi = np.arctan2(Rp[1, 0], Rp[0, 0])  # azimuth of the body x axis
        h = np.array([1.0, 0, 0]) - u[0] * u  # body x projected onto the horizontal plane (in body coordinates)
        n = np.linalg.norm(h)
        if n < 1e-3:
            return
        h /= n
        self.R = rotz(psi) @ np.stack([h, np.cross(u, h), u])  # rows = world axes expressed in the body frame


class Session:
    def __init__(self):
        self.clear()

    def clear(self):
        self.raw, self.t_last, self.att = [], None, Attitude()
        self.T, self.AB, self.AO, self.W, self.G = [], [], [], [], []

    def feed(self, d):
        self.raw.append(d)
        if self.t_last is not None and d["t"] > self.t_last:
            self.att.step(d["g"], d["w"], min(d["t"] - self.t_last, 0.1))
        self.t_last = d["t"]
        self.T.append(d["t"]); self.AB.append(d["ab"]); self.AO.append(d["ao"]); self.W.append(d["w"]); self.G.append(d["g"])

    def analysis(self, last=None):
        if len(self.T) < 10:
            return {"n": len(self.T), "duration": 0.0, "windows": 0}
        sl = slice(-int(last), None) if last else slice(None)
        return imu_check.analyse(np.array(self.T[sl]), np.array(self.AB[sl]), np.array(self.W[sl]))


# ----------------------------------------------------------------------------------------------- sample sources
def demo_samples():
    t, Rwb = imu_synth.trajectory(60.0, 50.0)
    raw = imu_synth.raw_from_truth(CALIB, t, Rwb, noise=0.02)
    blk, out = ImuBlock(CALIB), []
    for k in range(len(t)):
        x = blk.update(raw["a_raw"][k], raw["w_raw"][k], float(t[k]))
        out.append(dict(t=float(t[k]), ab=blk.ab.tolist(), ao=x[0:3].tolist(), w=x[3:6].tolist(), g=x[6:9].tolist()))
    return out


def replay_samples(path):
    return [json.loads(line) for line in open(path) if line.strip().startswith("{")]


def pump(samples, q, realtime=True):
    t_prev = None
    for d in samples:
        if realtime and t_prev is not None:
            time.sleep(max(0.0, min(d["t"] - t_prev, 0.2)))
        t_prev = d["t"]
        q.put(d)


def pump_live(q, port, host, cmd, procs):
    p = subprocess.Popen(["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, cmd],
                         stdout=subprocess.PIPE, text=True, bufsize=1)  # stderr (status lines) goes to this terminal
    procs.append(p)
    for line in p.stdout:
        if line.startswith("{"):
            q.put(json.loads(line))
    q.put(None)  # stream ended


# ----------------------------------------------------------------------------------------------- drawing
def box_faces(lo, hi):
    """Six quads of an axis-aligned box in the body frame: bottom, top, right(-y), left(+y), front(+x), rear."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return np.array([
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]])


HEX_ORDER = ("right_front", "right_middle", "right_back", "left_back", "left_middle", "left_front")


class View:
    def __init__(self, sess, src_name, full_mesh=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.plt, self.sess, self.src = plt, sess, src_name
        tri = load_stl()
        self.tri = tri if full_mesh else box_faces(tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0))
        self.legs = load_legs()
        pos = {n: (x, y) for n, x, y, _ in self.legs}
        self.hexa = np.array([[*pos[n], 0.0] for n in HEX_ORDER + (HEX_ORDER[0],)])  # footprint through the coxa mounts
        self.fig = plt.figure(figsize=(15, 7.5))
        self.ax = self.fig.add_axes([0.0, 0.02, 0.52, 0.96], projection="3d")
        ax = self.ax
        ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.2, 0.2); ax.set_zlim(-0.2, 0.2); ax.set_box_aspect((1, 1, 1), zoom=1.1)
        ax.view_init(elev=24, azim=-150)
        ax.set_xlabel("world x (start heading)"); ax.set_ylabel("world y"); ax.set_zlabel("up")
        fc = (0.55, 0.58, 0.68, 0.35) if full_mesh else [(0.6, 0.6, 0.65, 0.3), (0.6, 0.6, 0.65, 0.3), (0.6, 0.6, 0.65, 0.3),
                                                          (0.2, 0.75, 0.3, 0.45), (0.9, 0.25, 0.2, 0.55), (0.6, 0.6, 0.65, 0.3)]
        self.mesh = Poly3DCollection(list(self.tri), facecolors=fc, edgecolor=(0.15, 0.15, 0.25, 0.5), linewidths=0.6)
        ax.add_collection3d(self.mesh)
        self.hex_line = ax.plot(*self.hexa.T, color="navy", lw=1.5)[0]
        th = np.linspace(0, 2 * np.pi, 80)
        ax.plot(0.19 * np.cos(th), 0.19 * np.sin(th), 0 * th, color="0.7", lw=1)  # horizon ring
        ax.plot([-0.17, -0.17], [-0.17, -0.17], [0.05, -0.1], color="k", lw=3)  # gravity arrow, fixed in the world
        ax.text(-0.17, -0.17, -0.13, "gravity (down)", fontsize=9)
        self.leg_lines = [ax.plot([0, 0], [0, 0], [0, 0], color="k", lw=3)[0] for _ in self.legs]
        self.axes_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=4)[0] for c in ("r", "g", "b")]
        self.imu_pt = ax.plot([0], [0], [0], "o", color="m", ms=8)[0]
        self.front_txt = ax.text(0, 0, 0, "FRONT", color="r", fontsize=11, fontweight="bold")
        self.left_txt = ax.text(0, 0, 0, "LEFT", color="g", fontsize=11, fontweight="bold")
        self.panel = self.fig.text(0.52, 0.96, "", family="monospace", fontsize=8.8, va="top")
        self.r = np.array(json.load(open(CALIB)).get("lever_arm_body_m", [0, 0, 0]))
        self.last_check, self.items = 0.0, []

    def draw(self, now_checks=False):
        s = self.sess
        R = s.att.R
        self.mesh.set_verts(list(self.tri @ R.T))
        hx = self.hexa @ R.T
        self.hex_line.set_data_3d(hx[:, 0], hx[:, 1], hx[:, 2])
        for (name, x, y, yaw), ln in zip(self.legs, self.leg_lines):
            p0 = np.array([x, y, 0.0]); p1 = p0 + 0.09 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
            a, b = R @ p0, R @ p1
            ln.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
        for k, ln in enumerate(self.axes_lines):
            e = np.zeros(3); e[k] = 0.14
            v = R @ e
            ln.set_data_3d([0, v[0]], [0, v[1]], [0, v[2]])
        p = R @ self.r
        self.imu_pt.set_data_3d([p[0]], [p[1]], [p[2]])
        f, lf = R @ np.array([0.16, 0, 0.0]), R @ np.array([0, 0.16, 0.0])
        self.front_txt.set_position_3d((f[0], f[1], f[2])); self.left_txt.set_position_3d((lf[0], lf[1], lf[2]))
        if now_checks or time.monotonic() - self.last_check > 3.0:
            self.items = imu_check.verdict(s.analysis(last=60 * 50))
            self.last_check = time.monotonic()
        self.panel.set_text(self.text())
        self.fig.canvas.draw_idle()

    def text(self):
        s = self.sess
        if len(s.T) < 2:
            return "waiting for data...  (hold the robot still for the first 2 s)"
        g, w, ao, ab = np.array(s.G[-1]), np.array(s.W[-1]), np.array(s.AO[-1]), np.array(s.AB[-1])
        u = -g / np.linalg.norm(g)
        hz = 1.0 / max(1e-6, float(np.median(np.diff(s.T[-50:])))) if len(s.T) > 3 else 0.0
        head = np.degrees(np.arctan2(s.att.R[1, 0], s.att.R[0, 0]))
        L = ["IMU check  src=%s  %.0f Hz  n=%d   (q quit, r reset)" % (self.src, hz, len(s.T)), "",
             "nose-up  %+6.1f deg    left-up %+6.1f deg    heading %+6.1f deg (CCW +)" % (
                 np.degrees(np.arcsin(np.clip(u[0], -1, 1))), np.degrees(np.arcsin(np.clip(u[1], -1, 1))), head),
             "gyro  body  x %+6.2f  y %+6.2f  z %+6.2f rad/s" % tuple(w),
             "accel body  x %+6.2f  y %+6.2f  z %+6.2f   |a| %.2f m/s^2  (at sensor)" % (*ab, np.linalg.norm(ab)),
             "accel@orig  x %+6.2f  y %+6.2f  z %+6.2f" % tuple(ao),
             "gravity     x %+6.3f  y %+6.3f  z %+6.3f   (level = 0 0 -1)" % tuple(g), ""]
        for label, ok, detail in self.items:
            L.append("[%s] %-32s %s" % ({True: "OK", False: "  ", None: "??"}[ok], label[:32], detail[:46]))
        L += ["", "mesh must move like your hands move the robot:", "  front up <-> mesh front up, left up <-> mesh left up,",
              "  CCW from above <-> mesh turns CCW."]
        return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true"); g.add_argument("--demo", action="store_true")
    g.add_argument("--replay", metavar="FILE")
    ap.add_argument("--save-png", metavar="OUT", help="headless: process everything, save the last frame and exit")
    ap.add_argument("--mesh", action="store_true", help="draw the full STL triangle soup instead of the bounding box")
    ap.add_argument("--upto", type=float, help="with --save-png: stop feeding at this many seconds into the session")
    ap.add_argument("--port", type=int, default=2222); ap.add_argument("--host", default="pibot@127.0.0.1")
    ap.add_argument("--remote-cmd", default="python3 ~/imu_tools/tools/imu/imu_stream.py")
    ap.add_argument("--log", help="where to save the session (default tools/imu/logs/imu_<time>.jsonl)")
    a = ap.parse_args()
    src = "live" if a.live else "demo" if a.demo else "replay"
    sess = Session()

    if a.save_png:  # headless
        import matplotlib
        matplotlib.use("Agg")
        samples = demo_samples() if a.demo else replay_samples(a.replay)
        for d in samples:
            if a.upto is not None and d["t"] - samples[0]["t"] > a.upto:
                break
            sess.feed(d)
        v = View(sess, src, a.mesh)
        v.draw(now_checks=True)
        v.fig.savefig(a.save_png, dpi=100)
        print("wrote", a.save_png)
        print(imu_check.report(sess.analysis()))
        return

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.animation import FuncAnimation
    q, procs = queue.Queue(), []
    if a.live:
        th = threading.Thread(target=pump_live, args=(q, a.port, a.host, a.remote_cmd, procs), daemon=True)
    else:
        th = threading.Thread(target=pump, args=(demo_samples() if a.demo else replay_samples(a.replay), q), daemon=True)
    th.start()
    v = View(sess, src, a.mesh)
    v.fig.canvas.mpl_connect("key_press_event", lambda e: v.plt.close(v.fig) if e.key == "q" else (
        sess.clear() if e.key == "r" else None))

    def tick(_):
        try:
            while True:
                d = q.get_nowait()
                if d is None:
                    v.items = imu_check.verdict(sess.analysis()); break
                sess.feed(d)
        except queue.Empty:
            pass
        v.draw()

    anim = FuncAnimation(v.fig, tick, interval=60, cache_frame_data=False)  # noqa: F841
    # Tk swallows KeyboardInterrupt inside its event loop, so close the figure from a signal handler instead; the
    # animation timer gives Python a chance to run it. Ctrl-C (or SIGTERM from `timeout`) then ends the session cleanly.
    signal.signal(signal.SIGINT, lambda *_: v.plt.close("all"))
    signal.signal(signal.SIGTERM, lambda *_: v.plt.close("all"))
    v.plt.show()
    for p in procs:
        p.terminate()
    if sess.raw:
        path = a.log or os.path.join(HERE, "logs", time.strftime("imu_%Y%m%d_%H%M%S.jsonl"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(d, separators=(",", ":")) + "\n" for d in sess.raw)
        print("saved", path)
        print(imu_check.report(sess.analysis()))


if __name__ == "__main__":
    main()
