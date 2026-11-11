#!/usr/bin/env python3
"""Render the pxmark4 URDF base_link frame (what hexasim's IMU sits on) over the real base mesh.

Reads the leg origins straight from pxmark4.urdf.xacro and the base STL, so the picture is the URDF, not a redraw.
  python3 tools/imu/render_frame.py [out.png]
Axes: +x forward (red), +y left (green), +z up (blue). Origin = base_link origin. The IMU spot is the base_link origin.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from urdf_geom import base_com, legs as load_legs, load_stl

OUT = sys.argv[1] if len(sys.argv) > 1 else "base_link_frame.png"
tri = load_stl()  # metres
pts = tri.reshape(-1, 3)
print("base mesh extents (m) in base_link frame: x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f" % (
    pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(), pts[:, 2].min(), pts[:, 2].max()))
print("bbox centre (m): %s" % np.round((pts.min(0) + pts.max(0)) / 2, 4))
legs = load_legs()
com = base_com()
print("COM in base_link (m):", com)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.5, 1]})

# --- top view: x forward is drawn UP, y left is drawn LEFT (what you see looking down at the robot, nose at top)
def top(p):  # (x, y) -> plot (u = -y, v = x)
    return -p[..., 1], p[..., 0]

for t in tri[:: max(1, len(tri) // 6000)]:
    u, v = top(t[:, :2])
    ax.fill(u, v, color="0.75", lw=0.1, alpha=0.5)
for n, x, y, yaw in legs:
    u, v = top(np.array([x, y]))
    ax.add_patch(Circle((u, v), 0.012, color="k"))
    d = 0.09 * np.array([np.cos(yaw), np.sin(yaw)])
    du, dv = top(d)
    ax.annotate("", xy=(u + du, v + dv), xytext=(u, v), arrowprops=dict(arrowstyle="-", color="k", lw=2))
    ax.text(u * 1.28, v * 1.22 + (0.012 if v >= 0 else -0.02), n.replace("_", " "), ha="center", fontsize=9)
L = 0.10
for vec, c, lab in (((1, 0), "r", "+x forward"), ((0, 1), "g", "+y left")):
    du, dv = top(np.array(vec) * L)
    ax.annotate("", xy=(du, dv), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=3))
    ax.text(du * 1.15, dv * 1.1, lab, color=c, fontsize=12, fontweight="bold", ha="center")
ax.plot(0, 0, "o", color="b", ms=12)
ax.annotate("base_link origin (+z up, out of page)\n= IMU frame origin in sim", xy=(0, 0), xytext=(0.02, -0.075),
            color="b", fontsize=10, arrowprops=dict(arrowstyle="-", color="b"))
cu, cv = top(np.array(com[:2]))
ax.plot(cu, cv, "x", color="m", ms=10, mew=2)
ax.annotate("COM (%.1f, %.1f) mm: ~on origin" % (com[0] * 1e3, com[1] * 1e3), xy=(cu, cv), xytext=(-0.09, 0.06),
            color="m", fontsize=9, arrowprops=dict(arrowstyle="-", color="m"))
ax.set_aspect("equal"); ax.set_xlim(-0.32, 0.32); ax.set_ylim(-0.30, 0.30)
ax.set_xlabel("<- right        left ->  (plot axis = -y)"); ax.set_ylabel("rear   <-   x forward   ->  front")
ax.set_title("pxmark4 base_link, TOP view (nose up)"); ax.grid(alpha=0.3)

# --- side view: x forward to the right, z up
for t in tri[:: max(1, len(tri) // 6000)]:
    ax2.fill(t[:, 0], t[:, 2], color="0.75", lw=0.1, alpha=0.5)
for vec, c, lab in (((1, 0), "r", "+x"), ((0, 1), "b", "+z up")):
    ax2.annotate("", xy=(vec[0] * 0.08, vec[1] * 0.08), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=3))
    ax2.text(vec[0] * 0.09, vec[1] * 0.09, lab, color=c, fontsize=12, fontweight="bold")
ax2.plot(0, 0, "o", color="b", ms=10); ax2.plot(com[0], com[2], "x", color="m", ms=10, mew=2)
ax2.axhline(-0.033, color="0.4", ls="--", lw=1); ax2.text(-0.14, -0.03, "base_bottom (z=-33 mm)", fontsize=8)
ax2.set_aspect("equal"); ax2.set_xlim(-0.2, 0.2); ax2.set_ylim(-0.07, 0.12); ax2.grid(alpha=0.3)
ax2.set_title("SIDE view (nose right)"); ax2.set_xlabel("x forward (m)"); ax2.set_ylabel("z up (m)")
fig.tight_layout(); fig.savefig(OUT, dpi=110)
print("wrote", OUT)
