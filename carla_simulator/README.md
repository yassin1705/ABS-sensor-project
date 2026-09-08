# CARLA ABS simulator

Install the CARLA client and live-window dependencies in the Python 3.11
environment:

```powershell
.\carla_env\Scripts\python.exe -m pip install -r .\carla_simulator\requirements.txt
```

`carla_abs_simulator.py` generates fixed five-second CSV datasets.

`carla_live_driver.py` is the Pygame diagnostic interface. It owns CARLA
synchronous ticks, renders the chase camera beside the diagnostic panel,
estimates four wheel speeds, emulates ABS pulses and the ECU, and publishes
measurement batches to the local diagnostic API.

The Pygame setup screen selects all healthy sensors or one faulty wheel. During
the drive it displays four live wheel cards, health percentages, final model/SPC
decisions, selected-wheel details, and residual history. WASD or the arrow keys
drive, Space applies the handbrake, and Escape stops the session. The default
launcher uses one combined window. `start_diagnostic_platform_separate.cmd`
retains the earlier native-CARLA plus separate-dashboard setup as a fallback.

Use `start_carla_dashboard.cmd` from the project root. It starts CARLA, the API,
and Pygame together and removes all of those session processes when Pygame is
closed.
