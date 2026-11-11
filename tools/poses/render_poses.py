#!/usr/bin/env python3
"""PNG renders of the runner's poses from the URDF meshes (base, coxa, femur, tibia STLs) with full FK.

    python3 tools/poses/render_poses.py        # -> tools/poses/{sleep,plant,wide_stand,stand}.png + standup_sequence.png

Each pose: iso, side (nose right) and top (nose up) views on the ground plane, toes marked, belly height and toe
radius annotated. The body height is set so the lowest point (toe or belly bottom) touches z = 0. Poses come from
scripts/poses.py (the exact arrays the runner commands), meshes from the vendor URDF package.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tools", "imu"))

import urdf_geom  # noqa: E402
from joint_map import Contract  # noqa: E402
from leg_ik import LegKin  # noqa: E402
from poses import BELLY_Z, Motion, Poses  # noqa: E402

C = Contract()
K = LegKin(C.kin)
P = Poses(C)
MESH = {n: urdf_geom.link_stl(n) for n in ("base_link", "coxa_link", "femur_link", "tibia_link")}
MESH = {n: m[:: max(1, len(m) // (5000 if n == "base_link" else 1500))] for n, m in MESH.items()}  # decimate
COLOR = {"base_link": (0.55, 0.58, 0.62), "coxa_link": (0.35, 0.38, 0.42), "femur_link": (0.25, 0.45, 0.70),
         "tibia_link": (0.85, 0.50, 0.20)}
LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])


def world_tris(q):
    """-> triangles (n, 3, 3) in a ground frame (z = 0 ground), colors (n, 3), toes (6, 3), body height h."""
    toes = K.toe_positions(q)
    h = -min(toes[:, 2].min(), BELLY_Z)
    off = np.array([0, 0, h])
    tris, cols = [MESH["base_link"] + off], [np.tile(COLOR["base_link"], (len(MESH["base_link"]), 1))]
    for i in range(6):
        fr = urdf_geom.leg_link_frames(C.kin["hip"][i], C.kin["hip_yaw"][i], C.kin["femur_off"], C.kin["tibia_off"],
                                       q[i], q[6 + i], q[12 + i])
        for link, (R, p) in fr.items():
            m = MESH[link] @ R.T + p + off
            tris.append(m)
            cols.append(np.tile(COLOR[link], (len(m), 1)))
    return np.concatenate(tris), np.concatenate(cols), toes + off, h


def project(view):
    """-> (3x3 rotation world->camera rows: right, up, toward viewer)."""
    if view == "side":  # looking along +y (from the robot's right side), x to the right, z up
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
    if view == "top":  # looking down, nose (x) up, left (y) to the left
        return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    az, el = np.radians(-60), np.radians(28)  # iso: front-right, from above
    fwd = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.array([right, up, -fwd])


def draw(ax, q, view, title=None, annotate=True):
    tris, cols, toes, h = world_tris(q)
    Rv = project(view)
    cam = tris @ Rv.T
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nn > 0, nn, 1)
    shade = 0.45 + 0.55 * np.abs(n @ LIGHT)
    order = np.argsort(cam[:, :, 2].mean(1))
    ax.add_collection(PolyCollection(cam[order][:, :, :2], facecolors=np.clip(cols[order] * shade[order, None], 0, 1),
                                     edgecolors="none"))
    # ground
    g = np.array([[x, y, 0] for x in (-0.35, 0.35) for y in (-0.35, 0.35)])[[0, 1, 3, 2]] @ Rv.T
    if view != "top":
        ax.add_collection(PolyCollection([g[:, :2]], facecolors=(0.9, 0.9, 0.85), edgecolors="0.6", zorder=0))
    t = toes @ Rv.T
    ax.plot(t[:, 0], t[:, 1], "o", color="crimson", ms=4, zorder=5)
    r, z = K.legs_plane(q)
    if annotate:
        txt = "belly %.1f cm up\ntoe r %.1f cm, toe z %.1f cm" % (100 * (h + BELLY_Z), 100 * r.mean(), 100 * z.mean())
        if not np.allclose(r, r[0], atol=1e-4):
            txt = "belly %.1f cm up\ntoe r %s cm" % (100 * (h + BELLY_Z), np.round(100 * r, 1))
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
    if view == "side":
        ax.axhline(0, color="0.4", lw=0.8)
        ax.plot([-0.3, 0.3], [h + BELLY_Z] * 2, ":", color="0.3", lw=0.8)
    ax.set_aspect("equal")
    ax.autoscale_view()
    lim = {"side": ((-0.30, 0.30), (-0.02, 0.26)), "top": ((-0.34, 0.34), (-0.34, 0.34)),
           "iso": ((-0.34, 0.34), (-0.20, 0.28))}[view]
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or view, fontsize=9)


def pose_png(name, q, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, v in zip(axs, ("iso", "side", "top")):
        draw(ax, q, v, "%s - %s" % (name.upper(), v))
    deg = np.degrees(q)
    fig.suptitle("%s: coxa %.1f, femur %.1f, tibia %.1f deg (all legs)" % (name.upper(), deg[0], deg[6], deg[12])
                 if np.allclose(q[:6], q[0]) and np.allclose(q[6:12], q[6]) else name.upper(), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("wrote", out)


def sequence_png(out):
    """Side-view keyframes of the stand-up (the sit-down is the same strip read right to left)."""
    names = [s.name for s in P.core]
    keys = [("SLEEP", P.SLEEP), ("PLANT", P.core[0].q(1.0)), ("push-up 50%", P.core[1].q(0.5)),
            ("WIDE_STAND", P.core[1].q(1.0)), ("step RF (mid-swing)", P.core[2].q(0.5)),
            ("3 legs re-stepped", P.core[4].q(1.0)), ("6 legs re-stepped", P.core[7].q(1.0)),
            ("rise 50%", P.core[8].q(0.5)), ("STAND", P.STAND)]
    total = sum(s.duration for s in P.core)
    fig, axs = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 6.2))
    for j, (lab, q) in enumerate(keys):
        draw(axs[0, j], q, "side", lab, annotate=False)
        draw(axs[1, j], q, "iso", "", annotate=False)
        _, _, _, h = world_tris(q)
        axs[0, j].text(0.03, 0.95, "belly %.1f cm" % (100 * (h + BELLY_Z)), transform=axs[0, j].transAxes,
                       fontsize=8, va="top")
    fig.suptitle("Stand-up (after the ramp to SLEEP): %s - %.1f s at <= 1.2 rad/s. Sit-down = right to left."
                 % (" > ".join(n.replace("step_", "") for n in names), total), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print("wrote", out)


def main():
    for n, q in P.named.items():
        pose_png(n, q, os.path.join(HERE, n + ".png"))
    sequence_png(os.path.join(HERE, "standup_sequence.png"))
    assert len(Motion(P.core, C.dt).ticks()) > 0


if __name__ == "__main__":
    main()
