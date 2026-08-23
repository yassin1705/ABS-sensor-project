# CARLA ABS simulator

Install the CARLA client and live-window dependencies in the Python 3.11
environment:

```powershell
.\carla_env\Scripts\python.exe -m pip install -r .\carla_simulator\requirements.txt
```

`carla_abs_simulator.py` generates fixed five-second CSV datasets.

`carla_live_driver.py` is launched automatically by the diagnostic browser
dashboard. It owns CARLA synchronous ticks, keeps CARLA's native spectator
behind the vehicle, estimates four wheel speeds, emulates ABS pulses and the
ECU, and publishes measurement batches to the local diagnostic API.

Choose all healthy sensors or one faulty wheel in the browser before starting.
A small Pygame status window reports the controls and selected fault, while the
3D chase view remains in `CarlaUE4.exe`. WASD or the arrow keys drive, Space
applies the handbrake, and Escape stops the session. No RGB camera sensor is
created and only measurements are sent to the browser dashboard.
