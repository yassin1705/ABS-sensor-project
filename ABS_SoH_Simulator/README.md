# MATLAB ABS Simulation Environment

This folder contains the MATLAB simulation and data-collection environment used by the ABS sensor-diagnostic project. It models vehicle and wheel dynamics, active wheel-speed sensors, nominal measurement uncertainty, ECU speed reconstruction, and controlled sensor faults.

The simulator produces labelled time-series data for the forecasting, statistical process control (SPC), and fault-parameter models located in the parent repository.

> Generated CSV datasets are local artifacts and are intentionally excluded from Git.

## Simulation flow

```text
JSON scenario
     │
     ▼
driver commands ──► vehicle and wheel dynamics
                              │
                              ▼
                    four active ABS sensors
                              │
                              ▼
                 nominal errors + fault injection
                              │
                              ▼
                    ABS ECU reconstruction
                              │
                              ▼
                 labelled telemetry and manifest
```

Wheel identifiers are:

- `FL`: front left;
- `FR`: front right;
- `RL`: rear left;
- `RR`: rear right.

## Folder structure

| Path | Responsibility |
| --- | --- |
| `core/SimulationConfig.m` | Loads and validates JSON scenario configuration. |
| `core/ScenarioGenerator.m` | Converts named or segmented scenarios into time-dependent driver commands. |
| `core/VehicleDynamics.m` | Simulates vehicle motion, wheel rotation, tire slip, load transfer, and braking. |
| `core/WheelSensor.m` | Produces event-based ABS pulse edges from wheel rotation. |
| `core/SensorNetwork.m` | Coordinates the four wheel sensors. |
| `core/SensorErrorInjector.m` | Adds nominal latency/jitter and injects configured sensor faults. |
| `core/ABSECU.m` | Reconstructs wheel speeds from received pulse periods at the ECU sampling rate. |
| `core/SimulationController.m` | Orchestrates one complete simulation step. |
| `core/SimulationDataLogger.m` | Collects full telemetry and exports a timestamped CSV file. |
| `core/BrakingDatasetGenerator.m` | Generates healthy or intermittently faulty braking campaigns plus manifests. |
| `core/ExperimentDatasetGenerator.m` | Generates balanced physical-regime experiments. |
| `scenarios/` | Version-controlled JSON scenario definitions. |
| `app/SimulationApp.m` | Interactive keyboard-driven simulation interface. |
| `tests/` | Focused MATLAB regression tests. |
| `simulation_results/` | Generated datasets; ignored by Git. |

## Requirements

- MATLAB R2025b or a compatible release.
- A desktop session for `SimulationApp`.
- Sufficient free disk space before generating large campaigns.

The current implementation uses MATLAB language, table, JSON, graphics, and timer functionality. No Simulink model is required for the batch pipeline.

## Quick start

Start MATLAB from the repository root and run:

```matlab
cd ABS_SoH_Simulator
outputPath = run_simulation();
```

This loads `scenarios/scenario_reference.json`, performs the configured simulation, and writes a timestamped CSV file under `simulation_results/`.

The function also returns the in-memory result table:

```matlab
[outputPath, results] = run_simulation();
head(results)
```

To select a scenario, duration, and output directory explicitly:

```matlab
[outputPath, results] = run_simulation( ...
    "scenarios/scenario_reference.json", ...
    10.0, ...
    fullfile(tempdir, "abs_example"));
```

Using a temporary output directory is recommended for smoke checks.

## Interactive application

Launch the local driving interface with:

```matlab
cd ABS_SoH_Simulator
addpath app
SimulationApp
```

Controls:

- Up arrow: throttle;
- Down arrow: brake;
- Left/right arrows: steering;
- `R`: reset the simulation.

The interface is intended for visual exploration. Dataset generation uses the deterministic batch pipeline.

## Scenario configuration

`scenarios/scenario_reference.json` demonstrates a straight braking run with an intermittent loss on the front-left sensor. `scenarios/scenario_physics_dataset.json` is the base configuration for generated physical campaigns.

A scenario configuration contains these main sections:

| Section | Examples |
| --- | --- |
| `meta` | Experiment name, description, and random seed. |
| `simulation` | Time step and duration. |
| `vehicle` | Mass, wheel radius, track, wheelbase, road friction, and yaw inertia. |
| `dynamics` | Wheel inertia, tire-slip parameters, drag, load transfer, and lock thresholds. |
| `brakeSystem` | Maximum brake torque and front/rear distribution. |
| `scenario` | Initial speed and named or segmented driver commands. |
| `sensor` | Pulse edges per revolution for each wheel. |
| `sensorErrors` | Nominal latency and absolute/relative timing jitter. |
| `ecu` | ECU sample time and no-pulse timeout. |
| `faults` | Faulty wheel, type, activation interval, severity, and fault parameters. |

For dataset generation, `simulation.dt` must equal `ecu.sampleTime`. The supplied scenarios use 0.01 seconds, producing telemetry at 100 Hz.

## Physical-regime dataset

`ExperimentDatasetGenerator` creates controlled examples of normal rolling, sliding, wheel lock, and randomized braking. The following small example writes its output to the system temporary directory:

```matlab
cd ABS_SoH_Simulator
addpath core

generator = ExperimentDatasetGenerator( ...
    fullfile(pwd, "scenarios", "scenario_physics_dataset.json"), ...
    fullfile(tempdir, "abs_physics_example"), ...
    2026);

distribution = struct( ...
    "normal_rolling", 0.25, ...
    "wheel_lock", 0.25, ...
    "sliding", 0.25, ...
    "random", 0.25);

[outputPath, data, manifest] = generator.generate(distribution, 8, 3.0);
```

The seed makes scenario allocation and parameter sampling reproducible.

## Healthy braking dataset

`BrakingDatasetGenerator` creates a time-series CSV and a one-row-per-simulation manifest. This bounded example generates eight healthy simulations:

```matlab
cd ABS_SoH_Simulator
addpath core

generator = BrakingDatasetGenerator( ...
    fullfile(pwd, "scenarios", "scenario_physics_dataset.json"), ...
    fullfile(tempdir, "abs_healthy_example"), ...
    4200);

[datasetPath, manifestPath, manifest] = generator.generate( ...
    8, 2.0, ...
    ChunkSize=2, ...
    ProgressEvery=8);
```

Large datasets are written progressively in chunks instead of being retained entirely in memory.

## Faulty braking dataset

The provided campaign script generates 5,000 five-second simulations with one intermittently faulty sensor per scenario:

```matlab
cd ABS_SoH_Simulator
generate_faulty_braking_dataset_5000
```

The campaign uses:

- 100 Hz sampling for five seconds;
- stratified initial speeds from 40 to 100 km/h;
- deterministic balancing of faulty wheels and severity levels;
- fault starts between 1.5 and 2.0 seconds;
- fault durations between 1.5 and 2.0 seconds;
- severity levels of 0.25, 0.50, 0.75, and 1.00;
- a base edge-drop probability of 0.50.

This is a full data-generation run and can create a multi-gigabyte CSV file. Use the direct generator API with a small simulation count when validating changes.

Fault injection occurs at the pulse-edge level before ECU reconstruction. The model therefore observes the measurement consequences of missing sensor pulses rather than an artificially modified final speed value.

## Output data

The full simulation logger records metadata and driver/vehicle signals including:

- experiment name, scenario type, and seed;
- time, throttle, brake, and steering;
- vehicle speed, yaw rate, and longitudinal/lateral acceleration;
- nominal sensor-error configuration.

For every wheel, it also records:

- true, ground, peripheral, and ECU-reconstructed wheel speed;
- wheel angular acceleration and slip ratio;
- normal load, tire force, brake torque, and road friction;
- wheel-lock state;
- ECU validity and status;
- signal level, generated edges, and dropped edges;
- fault-active state, fault type, and severity.

Braking campaigns add fields such as `simulation_id`, requested/observed physical phenomenon, steering-wheel angle, and scenario parameters. Their companion manifest provides one compact record per simulation for splitting, stratification, and evaluation.

Fault labels and physical truth columns are intended only for supervised training and post-run evaluation. They must not be used as diagnostic inputs during inference.

## Tests

Run the complete focused MATLAB test folder without starting CARLA or training models:

```matlab
cd ABS_SoH_Simulator
addpath core
results = runtests("tests");
assert(all([results.Passed]));
```

The tests cover:

- configuration and vehicle stepping;
- healthy pulse generation;
- ECU period-based reconstruction;
- realistic sensor uncertainty;
- batch CSV generation and fault activation;
- braking campaign constraints and manifests;
- normal rolling, sliding, wheel lock, and load consistency.

Test-generated files are written under `tempdir` so the repository remains clean.

## Reproducibility notes

- Keep scenario JSON files under version control.
- Record the campaign seed with every experiment.
- Split training, validation, and test data by `simulation_id`, never by individual time window.
- Keep all windows from the same simulation in the same partition.
- Do not commit generated CSV files; regenerate them from the scenario and seed instead.

## Limitations

This simulator is a research environment. It is not a validated vehicle model, hardware-in-the-loop system, or safety-certified ABS implementation. Results must be confirmed in an independent simulation or physical validation environment before drawing automotive safety conclusions.
