# CARLA ABS simulator

Install the CARLA client and live-window dependencies in the Python 3.11
environment:

```powershell
.\carla_env\Scripts\python.exe -m pip install -r .\carla_simulator\requirements.txt
```

`carla_abs_simulator.py` generates fixed five-second CSV datasets.

`carla_live_driver.py` is the Pygame diagnostic interface. It owns CARLA
synchronous ticks, keeps CARLA's native spectator behind the vehicle, estimates
four wheel speeds, emulates ABS pulses and the ECU, and publishes measurement
batches to the local diagnostic API.

The Pygame setup screen selects all healthy sensors or one faulty wheel. During
the drive it displays four live wheel cards, health percentages, final model/SPC
decisions, selected-wheel details, and residual history. The 3D chase view stays
in `CarlaUE4.exe`. WASD or the arrow keys drive, Space applies the handbrake,
and Escape stops the session. No RGB camera sensor or browser is used.
