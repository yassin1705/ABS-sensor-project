% TEST_STEP1_4_ECU_PERIOD
% Verifie la chaine nominale : rotation -> fronts -> vitesse estimee ECU.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "core"));

cfg = SimulationConfig(fullfile(projectRoot, "scenarios", "scenario_reference.json"));
vd = VehicleDynamics(cfg.vehicle);
sensors = SensorNetwork(cfg.sensor);
ecu = ABSECU(cfg.ecu, cfg.sensor, cfg.vehicle.wheelRadius);

dt = 0.001;
duration = 1.0;
N = round(duration / dt);
constantSpeed = 20; % m/s, soit 72 km/h
throttle = constantSpeed / (100 / 3.6);

for k = 1:N
    command = DriverCommand(throttle, 0, 0, constantSpeed);
    truth = vd.step(command.toStruct(), dt);
    signals = sensors.step(truth);
    measurements = ecu.step(signals);
end

assert(measurements.valid(1), "La mesure ECU FL doit etre valide apres 1 s.");
assert(measurements.status(1) == "period", ...
    "A vitesse constante, la mesure ECU doit provenir de la periode.");
assert(abs(measurements.wheelSpeedMeasured(1) - truth.wheelSpeedTrue(1)) < 1e-9, ...
    "La vitesse ECU saine doit correspondre a la vitesse reelle constante.");

fprintf("Test ECU periode reussi : %.3f km/h (statut %s).\n", ...
    measurements.wheelSpeedMeasured(1) * 3.6, measurements.status(1));
