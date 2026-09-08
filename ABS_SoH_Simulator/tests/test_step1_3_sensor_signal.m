% TEST_STEP1_3_SENSOR_SIGNAL
% Verifie la transformation saine : VehicleState -> fronts ABS horodates.
% Aucun ECU ni defaut n'est utilise dans ce test.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "core"));

cfg = SimulationConfig(fullfile(projectRoot, "scenarios", "scenario_reference.json"));
vd = VehicleDynamics(cfg.vehicle);
sensors = SensorNetwork(cfg.sensor);

dt = 0.001;
duration = 1.0;
N = round(duration / dt);
constantSpeed = 20; % m/s, soit 72 km/h
throttle = constantSpeed / (100 / 3.6);

edgeCountFL = 0;
lastSignalFL = struct();
for k = 1:N
    command = DriverCommand(throttle, 0, 0, constantSpeed);
    truth = vd.step(command.toStruct(), dt);
    signals = sensors.step(truth);
    edgeCountFL = edgeCountFL + signals.FL.edgesThisStep;
    lastSignalFL = signals.FL;
end

expectedEdges = floor(truth.wheelAngleTotal(1) / ...
    (2*pi / cfg.sensor.edgesPerRevolution.FL));

assert(edgeCountFL == expectedEdges, ...
    "Le nombre de fronts FL doit correspondre a l'angle reel cumule.");
assert(lastSignalFL.healthy, "Le capteur V1 doit etre sain.");
assert(all(lastSignalFL.edgeTimes >= 0 & lastSignalFL.edgeTimes <= truth.time), ...
    "Les fronts doivent etre correctement horodates.");

fprintf("Test capteur sain reussi : %d fronts FL en %.3f s.\n", ...
    edgeCountFL, truth.time);
