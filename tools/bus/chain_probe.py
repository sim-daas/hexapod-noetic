#!/usr/bin/env python3
"""SYNC_READ chain-break probe (Pi, Python 3.8). Torque stays OFF; nothing is written.

Under policy load one servo sometimes stays silent in the SYNC_READ, and every id behind it in the list drops too
(logs 2026-11-11: the final missing set is always a suffix of the read order, the bytes received are an exact multiple
of 15, and the immediate retry usually recovers nothing). This tool reproduces the broken chain with a GHOST id (one
that no servo has) in the middle of the list, then measures:

  A  baseline: SYNC_READ of 18 positions (time, replies)
  B  ghost at position k: which replies arrive (expect the ids before the ghost only)
  C  after B, wait d ms, then re-read the ids behind the ghost: SYNC_READ vs single READs. Shows how long the servos
     that were waiting for the ghost stay deaf
  D  FAST SYNC READ (0x8A, X-series firmware >= 45): time, replies, and what a ghost does to it

    python3 tools/bus/chain_probe.py [--n 200]
"""
import argparse
import os
import struct
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import dxl_bus as dx  # noqa: E402
from joint_map import Contract  # noqa: E402

GHOST = 99


def ms(x):
    x = np.asarray(x) * 1e3
    return "p50 %5.2f p99 %5.2f max %5.2f ms" % (np.percentile(x, 50), np.percentile(x, 99), x.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyDXL")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    c = Contract()
    ids = [int(i) for i in c.ids]
    bus = dx.DxlBus(args.port)
    try:
        bus.torque_off_all()
        time.sleep(0.05)
        bus.ser.reset_input_buffer()

        # A ------------------------------------------------------------------------------------------------
        ts, ok = [], []
        for _ in range(args.n):
            t0 = time.monotonic()
            got = bus.sync_read(ids, dx.PRESENT_POSITION, 4)
            ts.append(time.monotonic() - t0)
            ok.append(len(got))
            time.sleep(0.005)
        print("A  SYNC_READ 18 x 4 B: %s, replies min %d mean %.2f" % (ms(ts), min(ok), np.mean(ok)))

        # B ------------------------------------------------------------------------------------------------
        k = 12
        lst = ids[:k] + [GHOST] + ids[k:]
        got = bus.sync_read(lst, dx.PRESENT_POSITION, 4)
        print("B  ghost at list position %d: %d replies; behind the ghost: %d/%d (bytes %d)"
              % (k, len(got), sum(i in got for i in ids[k:]), len(ids) - k, bus.last_sync[0]))

        # C ------------------------------------------------------------------------------------------------
        behind = ids[k:]
        print("C  re-read of the %d ids behind the ghost, d ms after the ghost read returned:" % len(behind))
        for d in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
            rs, rr = [], []
            for _ in range(20):
                bus.sync_read(lst, dx.PRESENT_POSITION, 4)  # break the chain (returns ~GAP_S after the last reply)
                time.sleep(d / 1e3)
                g = bus.sync_read(behind, dx.PRESENT_POSITION, 4)
                rs.append(len(g))
                time.sleep(0.01)
                bus.sync_read(lst, dx.PRESENT_POSITION, 4)
                time.sleep(d / 1e3)
                rr.append(sum(bus.read(i, dx.PRESENT_POSITION, 4) is not None for i in behind))
                time.sleep(0.01)
            print("   d %5.1f ms   SYNC_READ %5.2f/%d   single READs %5.2f/%d"
                  % (d, np.mean(rs), len(behind), np.mean(rr), len(behind)))

        # D ------------------------------------------------------------------------------------------------
        def fast(lst, n=4, timeout=0.010):
            bus.ser.reset_input_buffer()
            bus._send(dx.packet(dx.BROADCAST, 0x8A, struct.pack("<HH", dx.PRESENT_POSITION, n) + bytes(lst)))
            per = 4 + n  # ERR ID DATA CRC
            raw = bus._recv_gap(8 + per * len(lst), timeout, dx.GAP_S)
            return raw

        def parse_fast(raw, lst, n=4):
            """-> {id: value} for the blocks whose running CRC checks out."""
            out = {}
            if len(raw) < 8 or raw[:5] != b"\xff\xff\xfd\x00\xfe" or raw[7] != 0x55:
                return out
            pos = 8
            for i in lst:
                blk = raw[pos:pos + 4 + n]
                if len(blk) < 4 + n:
                    break
                crc = struct.unpack("<H", blk[-2:])[0]
                if crc != dx.crc16(raw[:pos + 2 + n]) or blk[1] != i:
                    break
                out[i] = struct.unpack("<i", bytes(blk[2:2 + n]))[0]
                pos += 4 + n
            return out

        raw = fast(ids)
        print("D  FAST SYNC READ raw %d bytes (expect %d): %s" % (len(raw), 8 + 8 * 18, raw[:24].hex()))
        ts, ok = [], []
        for _ in range(args.n):
            t0 = time.monotonic()
            g = parse_fast(fast(ids), ids)
            ts.append(time.monotonic() - t0)
            ok.append(len(g))
            time.sleep(0.005)
        ref = bus.sync_read(ids, dx.PRESENT_POSITION, 4)
        g = parse_fast(fast(ids), ids)
        agree = sum(abs(g.get(i, -9999) - ref.get(i, 9999)) <= 2 for i in ids)
        print("   18 x 4 B: %s, parsed replies min %d mean %.2f; values agree with SYNC_READ on %d/18"
              % (ms(ts), min(ok), np.mean(ok), agree))
        raw = fast(lst)
        g = parse_fast(raw, lst)
        print("   ghost at %d: raw %d bytes, parsed %d (prefix), behind ghost %d" % (
            k, len(raw), len(g), sum(i in g for i in ids[k:])))
        time.sleep(0.01)
        rs = []
        for d in (0.0, 1.0, 3.0):
            raw = fast(lst)
            time.sleep(d / 1e3)
            rs.append((d, len(bus.sync_read(behind, dx.PRESENT_POSITION, 4))))
            time.sleep(0.01)
        print("   SYNC_READ of the ids behind, d ms after a ghosted FAST read: %s" % rs)
    finally:
        bus.torque_off_all()
        bus.close()


if __name__ == "__main__":
    main()
