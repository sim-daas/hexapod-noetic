# Handoff: S1 XLS on the robot; next = collect the walking data properly (written 2026-11-14)

Session of 2026-11-13 evening / night, driven from the hexasim side. Details: `hardware_progress.md`, the two entries
"2026-11-13 (evening)" and "2026-11-13 (night)"; hexasim `research/phase2-progress.md` §2.4.4–2.4.5.

## State

- **Policy on the Pi:** `--policy s1xls` = hexasim `logs/rsl_rl/hexapod_s1/2026-11-12_09-36-47_s1_xls_seed42/model_7999.pt`
  (blind S1, trained on the *provisional* XL430 model). 1,125-D obs, XL430 tick, gait clock on, command slew limit
  3.5 rad/s taken from the contract. `s1h4` is still the default policy and is unchanged.
- **Checked offline:** Pi parity with hexasim's modules 7.9e-6; `tools/test_teleop_offline.py` 38 / 38 for both policies
  (`HEXAPOD_POLICY=s1xls` / default); clock and slew against a recording of the real hexasim env (`tools/sim/check_clock_slew.py`).
- **First robot runs (user at the robot):** three sessions, no fault, dropped reads ≤ 0.01 %, period p99 20.0 ms.
  Logs `~/hexapod_logs/2026-11-13_14-25-40` (hung), `_14-27-13` (floor), `_14-30-37` (floor, taken to be the run over ≈ 5 cm
  objects), on the Pi and the workstation.
- **Result:** the gait transfers (step rate 0.77 vs 0.80 Hz in sim, same joint travel, standing calmer than sim). Under
  load the servos track ≈ 1.5 × worse than the sim (femur / tibia 113 / 116 vs 77 mrad RMS) and then move faster than the
  slew limit (p99 4.1–4.4 vs 3.2 rad/s). That is the gap the XL430 model fit has to close.
- **Robot:** the last log (`_14-30-37`) ends with sit-down complete and torque off. Battery was at 10.4–10.9 V under the
  policy — charge it.
- Start-up latched-alarm check and stall monitor are in the runner and on the Pi (the bench handoff below says otherwise; stale).
- Everything above was committed in `e4cfa5b` (2026-11-14). The Present PWM logging below is not committed yet.
  **Sync the workstation tree to the Pi before the data session** (rsync per `HARDWARE_ACCESS.md`).

## What the first runs did NOT give (the reason for a second session)

1. Walking only at **0.05 m/s** (41 s) and 2.7 s at 0.075. No 0.10 / 0.15, no turn commands in any log.
2. **No measured speed.** Leg odometry from the logs says 0.043 m/s at 0.05 commanded (upper estimate: assumes no foot slip).
3. **No Present PWM while walking.** The `duty` column is the runner's estimate from the position error; it reads 1.00
   during swing, where back-EMF takes most of it. Without the real PWM the loaded torque law cannot be checked.
4. Runs were short and not labelled: which log is which surface was inferred from the IMU, not recorded.
5. Total robot mass (kitchen / luggage scale) — still not measured.

## Data session to run (user at the robot)

Before: battery full (≥ 11.8 V at rest), LF connectors reseated, `python3 tools/bench/servo_bench.py status`.
Runner: `python3 scripts/hexapod_teleop.py --policy s1xls --soft-start 1.0 --vx-max 0.20`. One runner start per run, so each
run is its own log; say the run label out loud at the start of the video and note the log stamp the runner prints at exit.

| run | command | length | notes |
|---|---|---|---|
| A | zero command | 30 s | flat floor |
| B1–B3 | v_x 0.05 | 3 × ≥ 20 s | tape marks every 0.5 m, side video |
| C1–C3 | v_x 0.10 | 3 × ≥ 20 s | same |
| D1–D3 | v_x 0.15 | 3 × ≥ 20 s | same |
| E | v_x 0.20 | 1 × 20 s | only if D is clean |
| F | turn on the spot, `q` then `e` at 0.2 rad/s | one full turn each | overhead video if possible |
| G | v_x 0.10 with `a` / `d` (v_y ±0.025) | 20 s | optional |

Per run write down: label, log stamp, video file, distance covered and time (or leave it to the video), surface, anything seen
(slip, scuffing, a leg lagging). Stop a run on any stall warning that repeats. Goal PWM stays 885; `--policy-pwm 619` only if
reads drop. Obstacles are not part of this session (S1 does not train on them).

## Open work, in order

1. **Runner: log Present PWM while walking — done offline 2026-11-14, timing on the Pi still to check.**
   `DxlBus.health_full()` reads 124..146 (23 bytes, stuffing-aware since PWM / load / velocity are signed) in place of
   the 3-byte voltage + temperature read, still one servo every 2nd step (each joint every 0.72 s). New log columns:
   `hk` (sim joint index read that step), `hp` (Present PWM / 885, signed), `hl` (Present Load / 1000), `hw` (Present
   Velocity, rad/s), all nan on steps without a read. `health()` is unchanged for the bench. Tests 40 / 40 (both
   policies). On the Pi, before the runs: `python3 tools/bus/bus_bench.py --with-policy --threads 2 --imu-thread` (now times `health_full`;
   expect ≈ +0.2 ms on the health line, step p99 should stay ≲ 16 ms).
2. Data session (table above), then pull: `rsync -az -e 'ssh -p 2222' pibot@127.0.0.1:hexapod_logs/ ~/hexapod_logs/`.
3. hexasim: fit `actuators/xl430.py` on the bench logs + the 0.05 m/s walking logs, test by replaying the logged `q_cmd`
   through the sim robot (tool not written), and use the 0.10 / 0.15 / turn runs as held-out data.
4. Retrain S0 → S1 on the fitted model (≈ 17 h per seed, `ubuntu2`) — plan to be agreed with the user first.
5. Commit this repo. `tools/bench/test_bench_offline.py` did not finish within 2 min on the workstation on 2026-11-13
   (a copy from an earlier session was still hanging); not investigated.
6. From the bench handoff, still open: reseat LF connectors, optional `drops --caps 885` rerun on a full battery.

---

# Handoff: XL430 bench session (2026-11-13)

The bench is complete: hung tests, mass calibration, stall and ground have all run and been analysed. The battery is
charging. Next is building hexasim's XL430 actuator model from the logs.

## Robot state at the end

- Sitting belly-down in the SLEEP pose, torque off on all 18 servos, all in position mode (3), **no latched alarms**
  (id 2's overload trips from the stall test were each cleared by reboot).
- Battery 11.35 V at rest when the session ended (it was 12.0 V before stall), hottest servo 42 C.
- The vendor xs_sdk is not running. Nothing is left in PWM mode.

## Runs (logs in `~/hexapod_logs/bench/` on the Pi **and** on the workstation)

| run | log stem | key result |
|---|---|---|
| speed RF coxa (PWM mode) | `11-48-01_speed_RF_coxa` | no-load 5.35 rad/s (51 rpm) at ~11.7 V; t63 ~75 ms; static friction between 8 and 12 % duty, coulomb ~1.8 % |
| step RF coxa/femur/tibia | `11-48-34`, `11-49-06`, `11-49-39` | dead time 13–40 ms (longest on tiny steps), ~20 % overshoot at 0.1–0.2 rad, final error 0.5–7 mrad |
| sine RF ×3 | `11-50-19`, `11-50-59`, `11-51-39` | tracks to ~3 Hz, resonant peak ~1.25 near 5 Hz, falls off between 8 and 12.5 Hz, ~50 ms lag |
| hold RF femur, no mass | `11-52-13_hold_RF_femur` | P-law measured/predicted 1.014, friction 2.4 % duty, 8–10 mrad gravity sag |
| drops, cap 885 | `11-53-00_drops` | **aborted at 9.9 V** at the start of the 0.20 rad level; 0.79 % drops at 0.15 rad |
| drops, cap 619 | `11-55-25_drops` | complete; 0 drops up to 0.10 rad, 0.24 % at 0.15, 0.14 % at 0.20; V min 10.1 |
| hold RF femur, 0.465 kg | `12-08-03_hold_RF_femur_mass` | **tau_stall 1.33 N·m at 11.2 V** (fit r² 0.93); 30–60 mrad sag under load |
| stall LB femur, partial | `13-34-30_stall_LB_femur` | only the first two trials (30 % timeout, 50 % trip 15.16 s); origin not confirmed, see below |
| stall LB femur, default trials | `13-44-57_stall_LB_femur` | 30 % no trip in 40 s; 50 % trip 15.2 s; the 70 % and 100 % trials only reached 56 % and 63 % duty (7.7 s, 4.2 s) because the foot gave way |
| stall LB femur, more weight | `13-51-45_stall_LB_femur` | caps saturated: **70 % → 4.18 s, 85 % → 4.16 s, 100 % → 4.12 s**; V min 10.7 |
| ground | `13-59-57_ground` | complete, no stall, no drops; stand hold ~14 % tibia duty; peak 35 % (stand-up); tilt max 7.2°; V min 10.9 |

All dates are 2026-11-13. `python3 tools/bench/analyze_bench.py` was run on the Pi after the ground test;
`bench_summary.json` (in the same directory) holds every run's summary and the torque calibration.

### Findings

1. **Torque is close to the datasheet** (1.33 N·m vs ~1.4 at 11.1 V), so the sim's torque level is roughly right.
2. **The battery sags under load**: 11.5 V at rest went below 10 V when all 18 servos reversed at once with
   0.2 rad steps at the 100 % cap. The 70 % cap kept it above 10.1 V. One stalled femur at 100 % took a
   part-charged pack (11.6 V at rest) down to 10.7 V.
3. **Read drops come from the left-front leg**: chain breakers were mostly id 9 (LF tibia) and 7 (LF coxa), sometimes
   8, and 18 (RB tibia). They start only when the voltage sags to about 10.5 V or below, which points at a marginal
   connector or supply cable on that branch. The ground test (V min 10.9) had zero drops. Reseat the LF connectors
   (and check the cable between LF and RB).
4. **The soft P gain shows up as sag**: with 0.44 N·m at the toe the femur sits 60 mrad below target. In the ground
   test the tracking error was 22–29 mrad in the static holds and 50–83 mrad while moving. The sim's XL430 model
   must reproduce this.
5. **Overload trip time vs duty** (LB femur, stalled against the floor, 39–43 C):

   | measured duty | time to overload trip |
   |---|---|
   | 30 % | none in 40 s |
   | 50 % | 15.2 s |
   | 56 % | 7.7 s |
   | 63 % | 4.2 s |
   | 70 % | 4.2 s |
   | 85 % | 4.2 s |
   | 100 % | 4.1 s |

   So the trip time falls steeply between 50 % and 63 % and is flat at about 4.2 s above that. The 70 % scripted
   cap therefore buys no extra time against a hard stall: the stall monitor has to act well inside 4 s.
6. **Standing needs little torque.** STAND hold: tibia ~14 % duty, femur ~4 %, coxa ~1.5 %. The 2026-11-11 policy
   stance loads LB most (LB tibia 27 %, LB femur 21 %, the rest ≤ 12 %) and gave identical numbers at the 70 % and
   100 % caps. Nothing in the ground test came near either cap.
7. **One overload trip is not explained.** The first mass run (sweep −0.4..0.4 rad, up to 0.65 N·m) tripped id 11
   overload at the start. The steady demand there is only ~55 % duty, under the 70 % cap, so the cause was probably a
   transient. With finding 5 in hand, 55 % duty should take roughly 8–15 s to trip, which supports a transient
   rather than the steady load. No data survived that run.

### Notes on the stall and ground runs

- The default stall requests (cap 70 % : request 100 %, 100 : 130) do not saturate the cap, because the foot moves
  90–130 mrad past the contact point as the body lifts. The rerun used
  `--trials 0.7:1.6,0.85:1.8,1.0:2.0` with more weight on the body. Use requests about 0.9–1.0 above the cap.
- `13-34-30_stall_LB_femur` holds only two trials and starts at 33 C, which matches the robot just after the first
  status check. Nobody confirmed who started it or why it stopped; the two complete runs supersede it.
- The ground test refuses to start unless every joint is within tolerance of SLEEP (femur ±0.6, tibia ±1.0 rad). The
  legs were lying stretched out after the stall test, so they were ramped to SLEEP under power first
  (`Bench.torque_up` → `ramp(q, Poses.SLEEP, rate=0.3)` → torque off, as a one-off script over ssh). `ground` has no
  option for this yet.

## Code changes (not committed)

- `tools/bench/servo_bench.py` `t_hold` and `t_stall`: the `finally` saves the run **before** the return
  ramp/hold, which now catches `Abort`. Offline tests pass 16/16. **Synced to the Pi** (md5 `1928b638…` on both).
- **Teleop runner: stall monitor and start-up alarm check** (`scripts/safety.py`, `scripts/hexapod_teleop.py`, 4 new
  tests in `tools/test_teleop_offline.py`; 38/38 pass on the workstation and the Pi):
  - `setup_servos` reads Hardware Error Status of all 18 servos and refuses to start on a latched alarm, naming the
    joint and pointing at `servo_bench.py status --clear`. It does not reboot servos itself.
  - `Safety` estimates each joint's duty from the P-only law, `min(|q_cmd − q| / 0.164 rad, Goal PWM cap)`, so there
    is no extra bus read. FAULT (freeze → sit, as every fault) when a joint holds ≥ 60 % for 1.0 s or ≥ 45 % for
    5.0 s; WARN at half those times. The per-joint estimate is logged each step as `duty`.
  - Checked against real logs: on the ground test the estimate matches the measured PWM (correlation 0.997, mean
    difference 0.3 % duty) and nothing trips; on `2026-11-11_13-01-51` (the real id 2 overload) it faults 2.9 s after
    policy on, 2.8 s before the servo tripped. It also faults on `2026-11-06_09-55-11`, where the policy's standing
    pose held LB femur at ~50 % duty for 5 s (the bench says that trips in ~15 s).
  - Not covered: a stall that persists during the sit-down itself is still ignored (faults in SIT are), so the servo
    can trip there. The monitor has not run on the robot yet.
- Still uncommitted: `tools/bench/` (the whole bench), `handoff.md` and the `hardware_progress.md` entries
  (2026-11-12 bench code, 2026-11-13 results).

### Known issues not fixed

- `dxl_bus.py`'s vendor-driver check matches any process whose command line contains `xs_sdk`, including an ssh
  one-liner such as `pgrep -af xs_sdk; ...`. Harmless, but it gives false positives.
- The step and sine `finally` blocks still run bus commands (cap, goals) before `finish()`. Those don't call
  `check()`, so they shouldn't re-raise. Ground turns torque off and then saves, which is fine.
- `analyze_bench.py` prints a NumPy "All-NaN slice" warning on the stall runs (harmless).
- `servo_bench.py ground` cannot move the legs to SLEEP itself (see the note above).

## Next steps

1. Build hexasim's XL430 actuator model (`actuators/xl430.py`) from `~/hexapod_logs/bench/bench_summary.json` and
   the raw `.npz` logs: speed/torque line (5.35 rad/s no-load, 1.33 N·m stall at ~11.2–11.7 V), friction, the
   ~15–20 ms dead time and 50 ms lag, the P-only sag, the overload trip curve, and voltage sag.
2. First teleop run with the stall monitor: watch for `near stall` warnings in the policy's standing pose (LB femur
   sat near 50 % duty there on 2026-11-06).
3. Reseat the LF connectors and check the LF–RB cable.
4. Optional, on a full battery: rerun `drops --caps 885` to fill the missing 0.20 rad level.
5. Commit `tools/bench/`, `handoff.md` and `hardware_progress.md`.

## How the runs were driven

Over ssh from the workstation (`ssh -p 2222 pibot@127.0.0.1`, tunnel per `HARDWARE_ACCESS.md`) with `--yes` and
`</dev/null`. `--yes` skips every prompt, **including "hang the weight now" and "remove the weight"**, so physical setup
has to be done before the command starts. Don't put `xs_sdk` anywhere in the ssh command line (see the known issue
above). Logs were pulled with
`rsync -az -e 'ssh -p 2222' pibot@127.0.0.1:hexapod_logs/bench/ ~/hexapod_logs/bench/`.
