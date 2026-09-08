% TEST_STEP1_5_SENSOR_ERRORS
% Verifie la chaine avec incertitudes nominales parametrees entre le
% capteur ideal et l'ECU.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "core"));

cfg = SimulationConfig(fullfile(projectRoot, "scenarios", "scenario_reference.json"));
vd = VehicleDynamics(cfg.vehicle);
sensors = SensorNetwork(cfg.sensor);
injector = SensorErrorInjector(cfg.sensorErrors);
ecu = ABSECU(cfg.ecu, cfg.sensor, cfg.vehicle.wheelRadius);

dt = 0.001;
duration = 1.0;
N = round(duration / dt);
constantSpeed = 20; % m/s
throttle = constantSpeed / (100 / 3.6);
allErrorsKmh = zeros(1,0);

for k = 1:N
    command = DriverCommand(throttle, 0, 0, constantSpeed);
    truth = vd.step(command.toStruct(), dt);
    cleanSignals = sensors.step(truth);
    alteredSignals = injector.step(cleanSignals, truth);
    measurements = ecu.step(alteredSignals);

    valid = measurements.valid & isfinite(measurements.wheelSpeedMeasured);
    if any(valid)
        errors = (measurements.wheelSpeedMeasured(valid) - ...
            truth.wheelSpeedTrue(valid)) * 3.6;
        allErrorsKmh = [allErrorsKmh, errors]; %#ok<AGROW>
    end
end

assert(~isempty(allErrorsKmh), "L'ECU doit produire des mesures valides.");
assert(any(abs(allErrorsKmh) > 0), ...
    "Le mode healthy_realistic doit produire une incertitude non nulle.");
assert(all(isfinite(allErrorsKmh)), "Les ecarts doivent rester finis.");

fprintf("Test incertitudes reussi : min=%.6f, max=%.6f km/h.\n", ...
    min(allErrorsKmh), max(allErrorsKmh));
