# Local ABS diagnostic platform

The platform reuses the existing trained GRU, CNN, MSP limits, saved MATLAB
scenarios, and generated CARLA scenarios. It does not retrain a model, rerun
MATLAB, or start CARLA when a scenario is loaded.

## API

- `GET /api/health`: process health.
- `GET /api/scenarios`: balanced healthy/faulty scenario catalog.
- `POST /api/diagnose`: runs both pipelines for one scenario.

Example body:

```json
{"source": "faulty", "simulation_id": 5}
```

CARLA datasets generated under `carla_simulator/simulation_results` use the
sources `carla_faulty` and `carla_healthy`:

```json
{"source": "carla_faulty", "simulation_id": 1}
```

The current models and MSP limits were learned from MATLAB data, so CARLA
results are reported as cross-domain evaluations.

## Final wheel decision

Raw SPC and independent CNN/GRU outputs remain visible, but the sensor card uses
an explainable decision layer:

- `HEALTHY`: neither layer indicates a fault;
- `WARNING`: SPC anomaly without an independent primary wheel;
- `CROSS_EFFECT`: SPC alarm caused by another wheel's established primary fault;
- `SUSPECTED_FAULTY`: highest independent probability, awaiting persistence or SPC support;
- `CONFIRMED_FAULTY`: persistent independent winner with SPC/localization support;
- `AMBIGUOUS`: several independent probabilities are high without a sufficient margin;
- `INSUFFICIENT_SIGNAL`: the five-second independent input is unusable.

Live isolation requires two matching candidates in the latest three rolling
evaluations. The winning probability must exceed 0.50 and lead the second wheel
by at least 0.10; otherwise the result remains ambiguous.

The dashboard replays the returned 500 samples at 100 Hz. The health percentage
is a documented prototype heuristic, not a calibrated failure probability.

## Live CARLA drive

The home dashboard also supports a live CARLA session:

1. Start the CARLA server directly on Town04 using the more stable D3D11 path:
   `CarlaUE4.exe /Game/Carla/Maps/Town04 -quality-level=Low -dx11 -windowed -ResX=960 -ResY=540`.
2. Start this platform with `start_diagnostic_platform.cmd`.
3. Select all-healthy sensors or one faulty wheel in the dashboard.
4. Press **Start live drive**. A separate CARLA/Pygame driving window opens.
5. Drive with WASD or the arrow keys; Space applies the handbrake and Escape
   stops the session.

### Combined CARLA camera and dashboard

For a simple native cockpit diagnostic view, start CARLA and then run
`start_carla_dashboard.cmd`. Select the sensor condition in the temporary
Pygame setup window and press Enter. The setup window closes, CARLA switches to
the cockpit spectator, and four wheel-health percentages appear in the 3D
view. The browser dashboard is not required.

The driving window remains local. Only compact 100 Hz ABS telemetry is sent to
the diagnostic API. GRU/SPC starts after its 20-sample warm-up, and CNN/GRU is
evaluated on a rolling 500-sample window every 50 new samples.

## Start locally

Double-click `start_diagnostic_platform.cmd` at the project root. The launcher
starts the Python API, the dashboard, and opens `http://localhost:3000`.
