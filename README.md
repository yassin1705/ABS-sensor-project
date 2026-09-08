# Intelligent Diagnosis of ABS Wheel-Speed Sensors

This end-of-studies project implements an experimental pipeline for detecting and characterizing ABS wheel-speed sensor faults. It combines a reproducible MATLAB simulation, dataset-generation tools, neural-model benchmarks, statistical process control (SPC), a local diagnostic service, and cross-domain integration with CARLA 0.9.16.

> This repository contains source code only. Generated datasets, trained model files, caches, experiment runs, reports, and presentations are intentionally excluded.

## Pipeline

```text
MATLAB ABS simulation
        │
        ├── healthy and faulty braking campaigns
        │             │
        │             ▼
        │    dataset preparation and split isolation
        │             │
        │             ├── wheel-speed forecasting: CNN / GRU / LSTM
        │             └── fault diagnosis: CNN / GRU / LSTM / CNN-GRU
        │                           │
        │                           ▼
        │                selected models + SPC limits
        │                           │
        └───────────────────────────┤
                                    ▼
                         diagnostic API and decision layer
                                    │
                                    ▼
                         CARLA integration and validation
```

The MATLAB domain is used to create controlled, labelled campaigns. CARLA reuses the same telemetry contract to evaluate the diagnostic pipeline in a second simulation domain. CARLA results must therefore be interpreted as cross-domain validation, not as safety certification.

## Repository structure

| Path | Responsibility |
| --- | --- |
| `ABS_SoH_Simulator/` | MATLAB vehicle, wheel, ABS sensor, ECU, fault-injection, scenario, logging, and dataset-generation environment. |
| `model_training/` | CNN, GRU, and LSTM benchmark for forecasting the four wheel speeds and producing residuals for SPC. |
| `fault_parameter_training/` | Healthy/faulty classification and estimation of fault start, duration, and severity. |
| `diagnostic_platform/` | Scenario loading, inference orchestration, SPC, wheel-isolation rules, live state, and local API. |
| `carla_simulator/` | CARLA-backed dataset generation and the live Pygame driving/diagnostic interface. |
| `start_*.cmd`, `start_*.ps1` | Windows launchers for the local demonstrations. |

Generated data and model artifacts remain in their component directories locally, but are ignored by Git.

## Requirements

- Windows 10 or 11 for the provided launchers.
- MATLAB R2025b or a compatible release for the MATLAB simulator.
- Python 3.11 for the combined training, diagnostic, and CARLA environment.
- CARLA 0.9.16 for live CARLA integration.
- A CUDA-capable GPU is optional; the training code can run on CPU.

CARLA itself is not included in this repository.

## Python setup

From the repository root, create the model/diagnostic environment:

```powershell
py -3.11 -m venv model_training\.venv
.\model_training\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CARLA, create a Python 3.11 environment and install the matching client dependencies:

```powershell
py -3.11 -m venv carla_env
.\carla_env\Scripts\python.exe -m pip install -r carla_simulator\requirements.txt
```

The combined launcher uses `model_training\.venv` by default. If another environment contains both the diagnostic and CARLA dependencies, point the launcher to it with `PFA_PYTHON_EXE`.

## Local artifact configuration

Datasets and model files are not published. Generate and train them locally, then configure their locations in the current PowerShell session as needed:

```powershell
$env:PFA_MATLAB_DATA_ROOT = "<directory-containing-MATLAB-datasets>"
$env:PFA_CARLA_DATA_ROOT = "<directory-containing-CARLA-datasets>"
$env:PFA_HEALTHY_DATASET = "<path-to-healthy-MATLAB-dataset.csv>"
$env:PFA_FORECAST_CHECKPOINT = "<path-to-GRU-best_model.pt>"
$env:PFA_FORECAST_METADATA = "<path-to-GRU-run_metadata.json>"
$env:PFA_SPLIT_CSV = "<path-to-simulation_splits.csv>"
$env:PFA_SPC_LIMITS = "<path-to-control_limits.json>"
$env:PFA_FAULT_EXPERIMENTS = "<directory-containing-cnn-and-gru-runs>"
$env:PFA_FAULT_CACHE = "<path-to-fault_parameter_dataset.npz>"
```

When an environment variable is omitted, the code uses the corresponding locally generated path inside this repository.

## 1. MATLAB simulation and data collection

Open MATLAB in `ABS_SoH_Simulator/` or run:

```matlab
cd ABS_SoH_Simulator
outputPath = run_simulation();
```

The default run uses `scenarios/scenario_reference.json`. The simulator exports 100 Hz telemetry with four true wheel speeds, four ECU measurements, validity flags, driver commands, vehicle state, fault labels, and dropped-edge counts.

The full faulty braking campaign is generated with:

```matlab
cd ABS_SoH_Simulator
generate_faulty_braking_dataset_5000
```

Outputs are written to `ABS_SoH_Simulator/simulation_results/` and are intentionally ignored by Git.

## 2. Forecasting benchmark and SPC

The notebook `model_training/notebooks/train_models.ipynb` prepares simulation-level train/validation/test partitions and compares CNN, GRU, and LSTM forecasters under the same input/output contract:

```text
[batch, 20 historical samples, 4 wheels]
    -> [batch, 5 future samples, 4 wheels]
```

The recorded experiment selected the GRU using validation performance. Its first-horizon residuals are used to calibrate four SPC control charts. See `model_training/README.md` for calibration and causal replay commands.

## 3. Fault-model benchmark and selection

Open the second benchmark with:

```powershell
.\start_fault_parameter_notebook.cmd
```

This pipeline compares temporal models on five-second, single-wheel sequences. It predicts fault probability and, conditionally for faulty signals, fault start, duration, and severity. Dataset splits are isolated by simulation to prevent leakage between train, validation, and test.

The current diagnostic runtime combines the trained CNN and GRU predictions. Experiment directories, checkpoints, notebook outputs, and caches remain local and are not distributed.

## Benchmark summary and selected models

The following values summarize the locally recorded validation-selected checkpoints. MAE and RMSE values for wheel-speed forecasting are expressed in m/s. Fault timing errors are expressed in seconds. Generated datasets, checkpoints, and full experiment histories are not included in this repository.

### Wheel-speed forecasting

| Model | Parameters | Best epoch | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 173,364 | 1 | 0.406 | 0.533 | 0.410 | 0.533 |
| **GRU** | **43,860** | **28** | **0.071** | **0.229** | **0.069** | **0.232** |
| LSTM | 56,660 | 29 | 0.101 | 0.233 | 0.101 | 0.237 |

The GRU is the selected wheel-speed forecaster because it achieved the lowest validation loss and validation MAE while also using fewer parameters than the CNN and LSTM. Test performance is reported only after validation-based selection. The selected GRU supplies the causal `t+1` predictions used by the SPC residual layer.

### Fault detection and parameter estimation

| Model | Parameters | Validation loss | Test accuracy | Test F1 | Start MAE | Duration MAE | Severity MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **GRU** | **109,124** | **0.0209** | 99.88% | **0.9957** | 0.122 | 0.126 | **0.0376** |
| CNN | 353,860 | 0.0211 | 99.88% | 0.9957 | **0.040** | **0.054** | 0.0468 |
| LSTM | 142,660 | 1.1542 | 31.52% | 0.2829 | 0.133 | 0.132 | 0.2511 |

The GRU is the single-model winner because it has the lowest validation loss, the best test F1 by a small margin, and the lowest severity error. The CNN nevertheless estimates fault start and duration substantially better. For that reason, the current runtime uses an explicit CNN–GRU fusion: it combines their fault probabilities, takes timing from the CNN, and takes severity from the GRU.

This fusion is an engineering selection based on complementary errors, not a demonstrated improvement over every individual model. A future validation campaign should add the fusion as its own benchmark row and evaluate it on an untouched CARLA test set.

## 4. Diagnostic platform

The local API exposes health, scenario, diagnosis, and live-telemetry endpoints. It combines:

- GRU next-step residuals and SPC persistence rules;
- independent CNN/GRU fault probabilities and parameter estimates;
- an explainable decision layer for healthy, warning, cross-effect, suspected, confirmed, ambiguous, and insufficient-signal states.

Component details are documented in `diagnostic_platform/README.md`.

## 5. CARLA integration

Set the CARLA installation directory before using a launcher:

```powershell
$env:CARLA_ROOT = "<CARLA-0.9.16-installation-directory>"
```

Optionally configure the Python executable and API port:

```powershell
$env:PFA_PYTHON_EXE = "<python-environment>\Scripts\python.exe"
$env:PFA_API_PORT = "8765"
```

Start the combined CARLA camera and diagnostic interface:

```powershell
.\start_carla_dashboard.cmd
```

Use the two-window fallback with:

```powershell
.\start_diagnostic_platform.cmd separate
```

CARLA dataset generation is available through `carla_simulator/carla_abs_simulator.py`. Generated CARLA datasets are written to `carla_simulator/simulation_results/` and ignored by Git.

## Validation status

The repository provides unit tests for MATLAB simulation, data preparation, SPC causality, fault-parameter encoding, decision logic, and model contracts. CARLA integration is operational, but a statistically representative CARLA validation campaign and aggregate cross-domain metrics remain future work.

Recommended CARLA validation metrics include detection precision/recall/F1, false-positive rate, wheel-localization accuracy, detection latency, and fault-parameter errors across healthy and faulty scenario seeds.

## Tests

Run the Python unit tests without training models:

```powershell
.\model_training\.venv\Scripts\python.exe -m unittest discover -s diagnostic_platform\tests -p "test_*.py" -v
.\model_training\.venv\Scripts\python.exe -m unittest discover -s model_training\tests -p "test_*.py" -v
.\model_training\.venv\Scripts\python.exe -m unittest discover -s fault_parameter_training\tests -p "test_*.py" -v
```

Run the MATLAB tests from MATLAB:

```matlab
cd ABS_SoH_Simulator
addpath core
results = runtests("tests");
assert(all([results.Passed]));
```

These checks do not start CARLA or perform full model training.

## Continuous integration

`.github/workflows/ci.yml` runs the lightweight Python unit tests and the MATLAB simulator tests on pushes and pull requests. CI does not download project datasets or checkpoints, start CARLA, or train models.

The MATLAB job uses MathWorks' GitHub Actions integration with MATLAB R2025b. Public repositories can use the hosted batch license; a private repository may require a MATLAB batch licensing token configured as the `MLM_LICENSE_TOKEN` repository secret.

## Reproducibility and data policy

- Random seeds and scenario definitions are stored in source-controlled configuration and code.
- Raw and processed datasets are not distributed.
- Trained checkpoints and exported models are not distributed.
- Reports and presentations are kept outside the published repository.
- Benchmark claims should be reproduced locally from newly generated data before comparison or deployment.

## Limitations

This is a research and educational prototype. It is not an automotive safety component, has not been validated on a physical vehicle, and must not be used to make safety-critical control decisions.

## License

The source code is released under the MIT License. See `LICENSE`.
