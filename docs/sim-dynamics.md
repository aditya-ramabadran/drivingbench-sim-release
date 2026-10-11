# Simulator step 2: dynamics twin

The Sandbox native stack running on a fitted Corolla. A tool call goes through the same request
envelope, `Bridge`, `Controller`, `LongControl`, `LatControlTorque` and Toyota torque limiter as on
the comma; only the car below openpilot is a model. Held out, it tracks the real car's speed,
wheel angle and path from the same tool calls about twice as well as replay's `VirtualNative`.

## Run

```sh
uv run python -m drivingbench.sim.series     # runs/artifacts/routes → runs/sim/series/<route>.npz (~20 s)
uv run python -m drivingbench.sim.fit        # → sim/params.json (~4 min, 8 processes)
uv run python -m drivingbench.sim.validate   # held-out route → runs/sim/validate/{summary,windows}.json, windows.png
```

`--routes` on `validate` scores other routes (`runs/sim/validate-train` holds the training-route
scores). Tests in `tests/test_sim_dynamics.py` need no route data.

| File | Role |
|---|---|
| `sim/series.py` | one 100 Hz table per openpilot route (carState ticks, latest of every other signal) plus the Sandbox tool calls placed on the log clock through `clocks` |
| `sim/plant.py` | the Corolla below openpilot: speed, wheel angle, pose (`Params`, `Corolla`) |
| `sim/fit.py` | fits `sim/params.json` on the training routes |
| `sim/car.py` | `SimNative`: the native stack on the plant, with the producer's `call`/`status` interface |
| `sim/validate.py` | replays logged tool calls through `SimNative` and `VirtualNative`, scores both |
| `sim/corolla.carparams` | the car's logged `CarParams` (VIN and firmware list removed) |

## What the logs established

- **Clock alignment.** Commands issued while moving change openpilot's desired curvature within
  10 ms of the logged tool time. From standstill the change waits for the launch (median 1.6 s),
  because openpilot steers only above 0.3 m/s.
- **Launch.** Positive acceleration requests start immediately; the car moves after 1.37 s
  (median of 97), or 3.25 s when Toyota's standstill hold is engaged (19). The hold engaged in 1 of
  164 engaged stops, and that time the operator was braking. It comes from operator braking or a
  disengaged stop, not from time stopped. Departure lurches at about 1.4 m/s² for 0.2 s.
- **Steering is torque-limited.** openpilot's torque ramps at Toyota's limit (0 to full in 1 s) and
  often saturates. The wheel turns fast at first, slows as the tire resists, sticks inside a
  friction band, and returns to center faster than it turns out.
- **openpilot reproduces exactly.** Re-running `LatControlTorque` and the Toyota limiter over the
  logged inputs reproduces the logged torques (correlation 0.9998, 99th-percentile error 0.02 of
  full scale). Torque parameters never changed (learned values = `CarParams`: lateral-acceleration
  factor 1.99, friction 0.196); lateral delay was 0.32 s in every route.
- **The bicycle model holds.** Speed × `VehicleModel` curvature from the measured wheel angle
  matches the gyro's yaw rate (1 s smoothing, correlation 0.999, slope 0.95 to 1.03) once
  reversing is accounted for: operators reversed between attempts, and logged speed is unsigned.
  At engaged angles it overstates yaw by 3 to 5%.
- **Settings never changed** across logged sessions: 180° steering scale, default speed control,
  3.5 m/s ceiling.

## Model

The simulator mirrors `controlsd.state_control` with the Bridge in place of the planner. Lateral
control is active only above 0.3 m/s. The car's motor torque reading for the Toyota limiter is
2,103 × applied torque, fitted (correlation 0.998). Toyota's one-frame steer-request cut above
100 deg/s is omitted.

The plant (`sim/plant.py`, parameters in `sim/params.json`):

- **Speed.** Acceleration follows `gain × actuators.accel + bias` through a delay and a first-order
  lag; speed is clamped at zero. At standstill a positive request must persist for the launch time
  before the car departs at the lurch acceleration. The operator's brake or a disengagement at a
  stop engages the hold, which lengthens the launch.
- **Steering wheel angle** \(\delta\), driven by applied torque \(\tau\) (−1…1) only while lateral
  control is active, otherwise held:
  \[ \dot\delta = g\,\mathrm{dz}(\tau - \delta/S,\ f),\qquad
     g = (g_\text{out}\ \text{or}\ g_\text{back}) + g_v v \]
  \(S\) is the angle where full torque balances the tire, \(f\) the friction band, \(\mathrm{dz}\) a
  deadzone. Fitted: \(S\) = 251°, \(f\) = 0.35, \(g_\text{out}\) = 24, \(g_\text{back}\) = 71,
  \(g_v\) = 58 per m/s.
- **Pose.** Yaw rate = speed × `VehicleModel.calc_curvature(angle)` with the logged learned steer
  ratio (14.8) and stiffness.

The speed parameters are tuned on closed-loop replay of the training routes; the steering
parameters on 3 s open-loop windows. The tuned speed values (0.04 s delay, 0.57 s lag, gain 2.7)
are effective, not physical: the open-loop fit gives a 0.77 s delay and gain 1. The linear response
cannot hold both the launch overshoot and the steady-speed behavior; see Gaps.

## Validation

Fit on the Sep 16 route and two smaller routes (19 engaged-moving minutes). Hold out the Sep 17
route: all 12 published trial sessions, 15 engaged-moving minutes, 218 windows.

Each accepted `set_motion` opens a window. Both models start from the logged state, including
openpilot's command history, and get the same tool calls at their logged times. A window ends at
30 s, the next session, or the first operator input, disengagement or gear change.

Truth:
- **Speed and wheel angle:** logged.
- **Heading:** integrated gyro, with the at-rest bias removed.
- **Position:** logged speed dead-reckoned along that heading. GPS is too coarse at 1 m/s.

Median absolute error, held out:

| | Speed m/s | Wheel angle ° | Heading ° | Position m | Windows |
|---|---|---|---|---|---|
| 5 s: simulator | **0.058** | **5.0** | **0.38** | **0.33** | 197 |
| 5 s: replay model | 0.069 | 19.4 | 1.48 | 0.34 | 197 |
| 10 s: simulator | **0.028** | **4.4** | **0.83** | **0.32** | 174 |
| 10 s: replay model | 0.045 | 18.0 | 3.21 | 0.41 | 174 |
| 30 s: simulator | **0.026** | **7.2** | **3.2** | **0.96** | 89 |
| 30 s: replay model | 0.038 | 22.3 | 9.3 | 1.87 | 89 |
| whole window RMS: simulator | **0.094** | **7.3** | **0.96** | **0.31** | 218 |
| whole window RMS: replay model | 0.117 | 21.5 | 3.2 | 0.52 | 218 |

The replay model (`replay/dynamics.py`) was fitted on Sep 17 segments, the held-out day, so this
comparison favors it. On the training routes the simulator's 30 s position error is 0.92 m against
2.35 m.

![Held-out windows: car, simulator, replay model](sim/dynamics-validation.png)

## Gaps

- **Speed.** The real car overshoots after launch: openpilot asks for deceleration while the car
  keeps accelerating for 1 to 2 s (up to 1.8 m/s on a 1.0 m/s target). It also holds speed with
  small positive requests (a deadband). A fitted nonlinear map overfit and made the closed loop
  worse; a constrained deadband-plus-creep model is the next thing to try.
- **Steering.** The full-torque holding angle varies from about 120° to 160° between drives; the
  fit uses one value (251° balance, about 150° in practice).
- **Not modeled.** Operator braking while moving, road grade and roll, and the one-frame steer cut.
- **Envelope.** Measured at 0.2 to 1.5 m/s with engaged steering. Faster driving uses the same
  equations untested.
