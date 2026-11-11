"""pxmark4 geometry straight from the vendor URDF and base mesh (shared by render_frame.py and imu_viz.py)."""
import re
import struct

import numpy as np

ROOT = "/home/ubuntu/gitrepos/interbotix_ros_crawlers/interbotix_ros_xshexapods/interbotix_xshexapod_descriptions"
URDF = ROOT + "/urdf/pxmark4.urdf.xacro"
STL = ROOT + "/meshes/pxmark4_meshes/pxmark4_base_link.stl"


def load_stl(path=STL, scale=0.001):
    """-> (n, 3, 3) triangle vertices in metres (the URDF applies scale 0.001 to this mesh)."""
    b = open(path, "rb").read()
    if b[:5] != b"solid" or b"facet" not in b[:300]:  # binary STL
        n = struct.unpack("<I", b[80:84])[0]
        rec = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        return rec["v"].astype(float) * scale
    v = np.array([[float(x) for x in m.groups()] for m in re.finditer(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", b.decode())])
    return v.reshape(-1, 3, 3) * scale


def link_stl(link):
    """Triangles (n, 3, 3) in metres of a pxmark4 link mesh in its own link frame: base_link, coxa_link, femur_link,
    tibia_link (visual origin is identity in the URDF for all of them)."""
    return load_stl(ROOT + "/meshes/pxmark4_meshes/pxmark4_%s.stl" % link)


def leg_link_frames(hip, hip_yaw, femur_off, tibia_off, qc, qf, qt):
    """FK chain of one leg in base_link: -> {link: (R 3x3, p 3)} for coxa_link, femur_link, tibia_link (URDF joints:
    coxa Rz at (x, y, 0, yaw), femur Ry at +femur_off, tibia Ry at +tibia_off)."""
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])

    R1, p1 = rz(hip_yaw + qc), np.asarray(hip, dtype=float)
    R2, p2 = R1 @ ry(qf), p1 + R1 @ np.asarray(femur_off)
    R3, p3 = R2 @ ry(qt), p2 + R2 @ np.asarray(tibia_off)
    return {"coxa_link": (R1, p1), "femur_link": (R2, p2), "tibia_link": (R3, p3)}


def _yaw(expr):  # "-pi/4", "-3*pi/4", "pi/2" ...
    m = re.fullmatch(r"(-?)(?:(\d+)\*)?pi/(\d+)", expr.strip())
    if not m:
        raise ValueError("unexpected yaw expression in URDF: %r" % expr)
    return (-1 if m.group(1) else 1) * float(m.group(2) or 1) * np.pi / float(m.group(3))


def legs(path=URDF):
    """-> [(name, x, y, yaw)] for each leg in base_link, as written in the URDF."""
    txt = open(path).read()
    return [(n, float(x), float(y), _yaw(yaw))
            for n, x, y, yaw in re.findall(r'prefix="(\w+)" x="([-\d.]+)" y="([-\d.]+)" yaw="\$\{([^}]*)\}"', txt)]


def base_com(path=URDF):
    txt = open(path).read()
    m = re.search(r'link name="\$\(arg robot_name\)/base_link">.*?<origin rpy="0 0 0" xyz="([^"]+)"/>\s*<mass', txt, re.S)
    return [float(v) for v in m.group(1).split()]
