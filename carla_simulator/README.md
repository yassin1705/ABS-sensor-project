# CARLA ABS simulator

Install the CARLA client and live-window dependencies in the Python 3.11
environment:

```powershell
.\carla_env\Scripts\python.exe -m pip install -r .\carla_simulator\requirements.txt
```

`carla_abs_simulator.py` generates fixed five-second CSV datasets.

`carla_live_driver.py` is launched automatically by either the web dashboard or
`start_carla_dashboard.cmd`. It owns CARLA synchronous ticks, estimates four
wheel speeds, emulates ABS pulses and the ECU, and publishes measurement batches
to the local diagnostic API.

The Pygame client is now the main operating window. Before driving, choose all
healthy sensors or one faulty wheel with 1-5 and press Enter. The window then
shows the CARLA RGB chase camera beside four live diagnostic cards. WASD or the
arrow keys drive, Space applies the handbrake, and Escape stops the session.
The camera remains local; only measurements are sent to the diagnostic API.

To run without the browser dashboard:

1. Start `CarlaUE4.exe` on Town04.
2. Run `start_carla_dashboard.cmd` from the project root.
