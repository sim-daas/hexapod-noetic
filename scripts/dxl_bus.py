"""DYNAMIXEL Protocol 2.0 over pyserial for the pxmark4's XL430-W250 servos (no vendor SDK). NumPy-free, Python 3.8.

Packets:  instruction  FF FF FD 00 id len_l len_h instr params.. crc_l crc_h      len = n_params + 3
          status       FF FF FD 00 id len_l len_h 55 error params.. crc_l crc_h   len = n_params + 4
CRC-16 (poly 0x8005, init 0). Byte stuffing (FF FF FD -> FF FF FD FD) is applied to instruction params. Status params
of positions (<= 4095), volts and temps cannot contain FF FF FD, so parse_status ignores stuffing; the signed
PWM / load / velocity block of health_full() can, and goes through parse_status_stuffed.

Per control step the runner uses one SYNC_READ of Present Position (all 18 replies, ~5.3 ms) and one SYNC_WRITE of Goal
Position. A servo whose reply is missing, late or corrupt is a dropped read (None). A silent servo also silences every id
behind it in the SYNC_READ list, so read_positions keeps the servos that broke the chain at the end of the list.
Only RAM registers are written.

Position: q = (pos - 2048) * 2*pi / 4096 (2048 = 0 rad, the Interbotix convention). The observation code re-quantises
q to the AX-12A tick the sim was trained with (proprio.py).
"""
import math
import struct
import time

try:
    import serial  # pyserial
except ImportError:  # offline tests on a machine without pyserial use FakeBus
    serial = None

TICK = 2.0 * math.pi / 4096.0
MODEL_XL430 = 1060

# Instructions
PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83
BROADCAST = 0xFE

# Control table (X-series, XL430-W250). EEPROM below 64: read only for this runner.
MODEL_NUMBER, FIRMWARE, ID, BAUD, RETURN_DELAY, DRIVE_MODE, OPERATING_MODE = 0, 6, 7, 8, 9, 10, 11
TEMP_LIMIT, MAX_VOLT, MIN_VOLT, PWM_LIMIT, VELOCITY_LIMIT, MAX_POSITION, MIN_POSITION, SHUTDOWN = (
    31, 32, 34, 36, 44, 48, 52, 63)
TORQUE_ENABLE, LED, STATUS_RETURN, HW_ERROR = 64, 65, 68, 70
POS_D, POS_I, POS_P, FF2, FF1 = 80, 82, 84, 88, 90
GOAL_PWM, GOAL_VELOCITY, PROFILE_ACCEL, PROFILE_VELOCITY, GOAL_POSITION = 100, 104, 108, 112, 116
PRESENT_PWM, PRESENT_LOAD, PRESENT_VELOCITY, PRESENT_POSITION = 124, 126, 128, 132
PRESENT_VOLTAGE, PRESENT_TEMP = 144, 146
RAM_START = TORQUE_ENABLE
GAP_S = 0.002  # SYNC_READ: silence that ends the reply stream (healthy replies arrive <= 1 ms apart via the FTDI)
OP_POSITION = 3

HW_ERROR_BITS = {0x01: "input voltage", 0x04: "overheating", 0x08: "motor encoder", 0x10: "electrical shock",
                 0x20: "overload"}
ALERT = 0x80  # status error byte: hardware error flagged (details in HW_ERROR)
ALARM_MASK = 0x01 | 0x04 | 0x08 | 0x10 | 0x20


def rad_to_pos(q):
    return int(min(4095, max(0, round(q / TICK) + 2048)))


def pos_to_rad(p):
    return (p - 2048) * TICK


def errors_str(hw):
    return ",".join(v for k, v in HW_ERROR_BITS.items() if hw & k) or "alert"


def crc16(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _stuff(params):
    out, i = bytearray(), 0
    p = bytes(params)
    while i < len(p):
        out.append(p[i])
        if p[i] == 0xFD and i >= 2 and p[i - 2] == 0xFF and p[i - 1] == 0xFF:
            out.append(0xFD)
        i += 1
    return bytes(out)


def packet(dxl_id, instr, params=b""):
    p = _stuff(params)
    body = bytes([0xFF, 0xFF, 0xFD, 0x00, dxl_id]) + struct.pack("<H", len(p) + 3) + bytes([instr]) + p
    return body + struct.pack("<H", crc16(body))


def le(value, n):
    return struct.pack("<B" if n == 1 else "<H" if n == 2 else "<i", value)


def parse_status(buf, n):
    """Split a byte stream of status packets with n params each -> {id: (error, params bytes)} (valid ones only)."""
    out, L, i = {}, 11 + n, 0
    while i + L <= len(buf):
        if buf[i:i + 4] != b"\xff\xff\xfd\x00":
            i += 1
            continue
        r = buf[i:i + L]
        if struct.unpack("<H", r[5:7])[0] == n + 4 and r[7] == 0x55 and crc16(r[:-2]) == struct.unpack("<H", r[-2:])[0]:
            out[r[4]] = (r[8], bytes(r[9:9 + n]))
            i += L
        else:
            i += 1
    return out


def unstuff(p):
    return p.replace(b"\xff\xff\xfd\xfd", b"\xff\xff\xfd")


def parse_status_stuffed(buf):
    """-> {id: (error, params)} for every valid status packet in buf, length from the header, stuffing removed.
    Needed for signed reads: e.g. load -1 (FF FF) followed by a velocity byte FD makes the servo insert an extra FD."""
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
        if pk[7] == 0x55 and crc16(pk[:-2]) == struct.unpack("<H", pk[-2:])[0]:
            out[pk[4]] = (pk[8], unstuff(bytes(pk[9:-2])))
            i = end
        else:
            i += 1
    return out


PWM_FULL = 885  # Goal / Present PWM units at 100 % duty
VEL_UNIT = 0.229 * 2.0 * math.pi / 60.0  # rad/s per Present Velocity unit
HEALTH_ADDR, HEALTH_N = PRESENT_PWM, 23  # 124..146: PWM, Load, Velocity, Position, VelTraj, PosTraj, Voltage, Temp


def unpack_health(p):
    """23 param bytes from 124 -> (volts, deg C, pwm duty, load, rad/s): duty = Present PWM / 885 (signed),
    load = Present Load / 1000 (signed fraction of stall torque, the firmware's current-based estimate)."""
    pwm, load, vel, _pos, _vt, _pt, volt, temp = struct.unpack("<hhiiIIHB", p)
    return volt / 10.0, float(temp), pwm / float(PWM_FULL), load / 1000.0, vel * VEL_UNIT


class BusError(Exception):
    pass


def vendor_processes():
    """Running Interbotix xs_sdk nodes: a second master on the half-duplex bus corrupts our packets. The Pi's vendor
    autostart (xshexapod_rpi4_boot.service) starts one at boot; it was disabled on 2026-11-06."""
    import os
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
        except (IOError, OSError):
            continue
        if "xs_sdk" in cmd:
            found.append("%s %s" % (pid, cmd[:80]))
    return found


class DxlBus:
    def __init__(self, port="/dev/ttyDXL", baud=1000000, timeout=0.003, sync_timeout=0.010):
        """timeout: s for one status packet; sync_timeout: s for all replies of one SYNC_READ."""
        if serial is None:
            raise BusError("pyserial not installed")
        others = vendor_processes()
        if others:
            raise BusError("vendor servo driver running on the bus, stop it first "
                           "(sudo systemctl stop xshexapod_rpi4_boot.service): %s" % "; ".join(others))
        self.ser = serial.Serial(port, baud, timeout=timeout, write_timeout=0.05, exclusive=True)
        self.timeout, self.sync_timeout = timeout, sync_timeout
        self.last_error = {}  # id -> status error byte with the ALERT bit set
        self.n_tx = self.n_drop = self.n_retry = 0
        self.order, self.breaks = None, {}  # adaptive SYNC_READ order (read_positions)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ low level
    def _send(self, pkt):
        if self.ser.in_waiting:
            self.ser.reset_input_buffer()
        self.ser.write(pkt)

    def _recv(self, n_bytes, timeout):
        self.ser.timeout = timeout
        return self.ser.read(n_bytes)

    def _recv_gap(self, n_bytes, timeout, gap):
        """Read up to n_bytes; stop at the deadline or after ``gap`` s with no new bytes once some have arrived. A broken
        SYNC_READ chain goes silent, so this returns ~gap after the last reply instead of waiting out the timeout.
        (pyserial's inter_byte_timeout maps to termios VTIME, 0.1 s granularity: useless here.)"""
        import os
        import select
        fd = self.ser.fd
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while len(buf) < n_bytes:
            now = time.monotonic()
            wait = deadline - now if not buf else min(deadline - now, gap)
            if wait <= 0 or not select.select([fd], [], [], wait)[0]:
                break
            buf += os.read(fd, n_bytes - len(buf))
        return bytes(buf)

    def _txrx(self, dxl_id, instr, params, n_reply):
        self.n_tx += 1
        self._send(packet(dxl_id, instr, params))
        r = parse_status(self._recv(11 + n_reply, self.timeout), n_reply).get(dxl_id)
        if r is None:
            self.n_drop += 1
            self.ser.reset_input_buffer()
            return None
        if r[0] & ALERT:
            self.last_error[dxl_id] = r[0]
        return r[1]

    def ping(self, dxl_id):
        """-> model number or None."""
        r = self._txrx(dxl_id, PING, b"", 3)
        return None if r is None else struct.unpack("<H", r[:2])[0]

    def read(self, dxl_id, addr, n):
        """-> int (little-endian, signed for 4 bytes) or None."""
        r = self._txrx(dxl_id, READ, struct.pack("<HH", addr, n), n)
        if r is None:
            return None
        return struct.unpack("<B" if n == 1 else "<H" if n == 2 else "<i", r)[0]

    def write(self, dxl_id, addr, value, n):
        """Single write, acknowledged. RAM only."""
        if addr < RAM_START:
            raise BusError("refusing EEPROM write (addr %d)" % addr)
        params = struct.pack("<H", addr) + le(value, n)
        if dxl_id == BROADCAST:
            self._send(packet(dxl_id, WRITE, params))
            return True
        return self._txrx(dxl_id, WRITE, params, 0) is not None

    def sync_write(self, addr, n, values_by_id):
        if addr < RAM_START:
            raise BusError("refusing EEPROM write (addr %d)" % addr)
        params = struct.pack("<HH", addr, n) + b"".join(bytes([i]) + le(v, n) for i, v in values_by_id)
        self._send(packet(BROADCAST, SYNC_WRITE, params))

    def sync_read(self, ids, addr, n):
        """-> {id: int} for the servos that answered (missing = dropped)."""
        self.n_tx += len(ids)
        self._send(packet(BROADCAST, SYNC_READ, struct.pack("<HH", addr, n) + bytes(ids)))
        raw = self._recv_gap((11 + n) * len(ids), self.sync_timeout, GAP_S)
        got = parse_status(raw, n)
        out = {}
        fmt = "<B" if n == 1 else "<H" if n == 2 else "<i"
        n_alert = 0
        for i in ids:
            r = got.get(i)
            if r is None:
                continue
            if r[0] & ALERT:
                self.last_error[i] = r[0]
                n_alert += 1
            out[i] = struct.unpack(fmt, r[1])[0]
        self.last_sync = (len(raw), len(out), n_alert)  # diagnostics: bytes received, replies parsed, alert bits
        self.n_drop += len(ids) - len(out)
        if len(out) < len(ids):
            self.ser.reset_input_buffer()
        return out

    # ------------------------------------------------------------------ helpers used by the runner
    def read_order(self, ids):
        """SYNC_READ id order: the servos that broke the chain most often go last (stable otherwise)."""
        if self.order is None or sorted(self.order) != sorted(ids):
            self.order, self.breaks = list(ids), {}
        return self.order

    def _chain_broke(self, culprit):
        b = self.breaks
        b[culprit] = b.get(culprit, 0.0) + 1.0
        if b[culprit] > 50.0:  # keep adapting: halve all counts
            for k in b:
                b[k] *= 0.5
        pos = {i: k for k, i in enumerate(self.order)}
        self.order.sort(key=lambda i: (b.get(i, 0.0), pos[i]))

    def read_positions(self, ids):
        """-> list of rad or None per id (same order as ids).

        Under load the SYNC_READ chain breaks: each servo answers after the previous id in the list, so one servo
        that stays silent silences every id behind it (logs 2026-11-11: the missing set is always a suffix of the
        list). The list is therefore ordered so that the servos which broke the chain most often come last
        (read_order); a break then costs the culprit alone. The missing ids are asked once more in a second, short
        SYNC_READ, the culprit last."""
        order = self.read_order(ids)
        got = self.sync_read(order, PRESENT_POSITION, 4)
        rx1, ok1, alert1 = self.last_sync
        missing = [i for i in order if i not in got]
        # bytes, replies (1st read), recovered by the retry, alerts, culprit id (0 = none)
        self.last_read_diag = (rx1, ok1, 0, alert1, 0)
        if missing:
            self.n_retry += 1
            self.n_drop -= len(missing)  # count a read as dropped only if the retry fails too
            self.n_tx -= len(missing)
            t, self.sync_timeout = self.sync_timeout, 0.0015 + 0.0005 * len(missing)
            retry = missing[1:] + missing[:1]
            try:
                got.update(self.sync_read(retry, PRESENT_POSITION, 4))
            finally:
                self.sync_timeout = t
            self.last_read_diag = (rx1, ok1, sum(i in got for i in missing), alert1, missing[0])
            self._chain_broke(missing[0])
        return [pos_to_rad(got[i]) if i in got and 0 <= got[i] <= 4095 else None for i in ids]

    def write_goals(self, ids, q):
        self.sync_write(GOAL_POSITION, 4, [(i, rad_to_pos(v)) for i, v in zip(ids, q)])

    def set_pwm(self, ids, value):
        """Goal PWM = the X-series torque cap (<= PWM Limit, 885 = 100 %)."""
        self.sync_write(GOAL_PWM, 2, [(i, int(value)) for i in ids])

    def torque(self, ids, on):
        self.sync_write(TORQUE_ENABLE, 1, [(i, 1 if on else 0) for i in ids])

    def torque_off_all(self):
        """Best effort, never raises."""
        for _ in range(2):
            try:
                self.write(BROADCAST, TORQUE_ENABLE, 0, 1)
                self.sync_write(TORQUE_ENABLE, 1, [(i, 0) for i in range(1, 19)])
                time.sleep(0.002)
            except Exception:
                pass

    def health(self, dxl_id):
        """-> (volts, deg C) or None (Present Input Voltage + Present Temperature, one 3-byte read)."""
        r = self._txrx(dxl_id, READ, struct.pack("<HH", PRESENT_VOLTAGE, 3), 3)
        return None if r is None else (struct.unpack("<H", r[:2])[0] / 10.0, float(r[2]))

    def health_full(self, dxl_id):
        """-> (volts, deg C, pwm duty, load, rad/s) or None: one 23-byte READ of 124..146 (Present PWM through Present
        Temperature), ~0.2 ms longer than health(). The reply may be byte-stuffed (signed fields), so it is read by
        its header length."""
        self.n_tx += 1
        self._send(packet(dxl_id, READ, struct.pack("<HH", HEALTH_ADDR, HEALTH_N)))
        buf = self._recv(11 + HEALTH_N, self.timeout)
        if len(buf) >= 7 and buf[:4] == b"\xff\xff\xfd\x00":
            need = 7 + struct.unpack("<H", buf[5:7])[0] - len(buf)  # stuffed reply: a few bytes longer
            if 0 < need <= 8:
                buf += self._recv(need, self.timeout)
        r = parse_status_stuffed(buf).get(dxl_id)
        if r is None or len(r[1]) != HEALTH_N:
            self.n_drop += 1
            self.ser.reset_input_buffer()
            return None
        if r[0] & ALERT:
            self.last_error[dxl_id] = r[0]
        return unpack_health(r[1])

    def hw_error(self, dxl_id):
        return self.read(dxl_id, HW_ERROR, 1)


class FakeBus:
    """Offline stand-in: each servo tracks its goal at a bounded speed when torque is on; reads can be dropped.
    drop_fn(step, id) -> True drops that read. q0: initial joint angles by id (rad)."""

    def __init__(self, q0_by_id, speed=4.0, dt=0.02, volts=12.0, temp=40.0):
        self.q = dict(q0_by_id)
        self.goal = dict(q0_by_id)
        self.on = {i: False for i in self.q}
        self.speed, self.dt = speed, dt
        self.volts, self.temp = volts, temp
        self.drop_fn = None
        self.dead = False
        self.step_n = 0
        self.writes = []
        self.pwm = {i: 885 for i in self.q}
        self.last_error = {}
        self.n_tx = self.n_drop = 0
        self.regs = {i: {MODEL_NUMBER: MODEL_XL430, RETURN_DELAY: 0, OPERATING_MODE: OP_POSITION, PWM_LIMIT: 885,
                         MIN_POSITION: 0, MAX_POSITION: 4095, POS_P: 640, POS_I: 0, POS_D: 4000}
                     for i in self.q}

    def close(self):
        pass

    def advance(self):
        self.step_n += 1
        for i in self.q:
            if self.on[i]:
                d = self.goal[i] - self.q[i]
                m = self.speed * self.dt
                self.q[i] += max(-m, min(m, d))

    def ping(self, i):
        return None if self.dead or i not in self.q else MODEL_XL430

    def read(self, i, addr, n):
        if self.dead:
            return None
        if addr == PRESENT_POSITION:
            return rad_to_pos(self.q[i])
        return self.regs[i].get(addr, 0)

    def read_positions(self, ids):
        out = []
        for i in ids:
            self.n_tx += 1
            if self.dead or (self.drop_fn and self.drop_fn(self.step_n, i)):
                self.n_drop += 1
                out.append(None)
            else:
                out.append(pos_to_rad(rad_to_pos(self.q[i])))
        return out

    def write(self, i, addr, value, n):
        if addr < RAM_START:
            raise BusError("refusing EEPROM write")
        for k in (self.q if i == BROADCAST else [i]):
            if addr == TORQUE_ENABLE:
                self.on[k] = bool(value)
            self.regs[k][addr] = value
        return not self.dead

    def sync_write(self, addr, n, values_by_id):
        if addr < RAM_START:
            raise BusError("refusing EEPROM write")
        self.writes.append((addr, list(values_by_id)))
        for i, v in values_by_id:
            if addr == GOAL_POSITION:
                self.goal[i] = pos_to_rad(v)
            elif addr == TORQUE_ENABLE:
                self.on[i] = bool(v)
            elif addr == GOAL_PWM:
                self.pwm[i] = v
            self.regs[i][addr] = v

    def write_goals(self, ids, q):
        self.sync_write(GOAL_POSITION, 4, [(i, rad_to_pos(v)) for i, v in zip(ids, q)])

    def set_pwm(self, ids, value):
        self.sync_write(GOAL_PWM, 2, [(i, int(value)) for i in ids])

    def torque(self, ids, on):
        self.sync_write(TORQUE_ENABLE, 1, [(i, 1 if on else 0) for i in ids])

    def torque_off_all(self):
        for i in self.q:
            self.on[i] = False

    def health(self, i):
        return None if self.dead else (self.volts, self.temp)

    def health_full(self, i):
        """P-only estimate of the duty from the goal error (gain of regs[POS_P]), capped at Goal PWM."""
        if self.dead:
            return None
        if not self.on[i]:
            return self.volts, self.temp, 0.0, 0.0, 0.0
        cap = self.pwm[i] / float(PWM_FULL)
        duty = (self.goal[i] - self.q[i]) / TICK * self.regs[i].get(POS_P, 640) / 128.0 / PWM_FULL
        duty = max(-cap, min(cap, duty))
        d = self.goal[i] - self.q[i]
        vel = 0.0 if abs(d) < 1e-9 else math.copysign(min(self.speed, abs(d) / self.dt), d)
        return self.volts, self.temp, duty, duty, vel

    def hw_error(self, i):
        return 0
