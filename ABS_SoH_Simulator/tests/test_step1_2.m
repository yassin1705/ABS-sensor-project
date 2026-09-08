% TEST_STEP1_2  Verification manuelle de la chaine Config -> DriverCommand -> VehicleDynamics
% Peut etre execute directement depuis tests/ (les chemins sont resolus
% relativement a l'emplacement de ce script, pas au repertoire courant).

projectRoot = fileparts(fileparts(mfilename("fullpath")));  % ABS_SoH_Project/
addpath(fullfile(projectRoot, "core"));

% 1) Chargement de la configuration (le scenario vit dans scenarios/, pas config/)
cfg = SimulationConfig(fullfile(projectRoot, "scenarios", "scenario_reference.json"));
rng(cfg.getSeed());
fprintf("Config chargee : %s\n", cfg.meta.description);

% 2) Initialisation du modele vehicule
vd = VehicleDynamics(cfg.vehicle);

% 3) Boucle de simulation simplifiee (sans capteurs/ECU pour l'instant)
dt = cfg.simulation.dt;
N  = cfg.numSteps();

speedLog = zeros(N,1);
timeLog  = zeros(N,1);

initSpeed_ms = cfg.scenario.initialSpeed_kmh / 3.6;

for k = 1:N
    t = (k-1) * dt;

    if t < cfg.scenario.brakeStartTime
        cmd = DriverCommand(0, 0, 0, initSpeed_ms);
    else
        cmd = DriverCommand(0, cfg.scenario.brakeCommand, 0, initSpeed_ms);
    end

    state = vd.step(cmd.toStruct(), dt);

    timeLog(k)  = state.time;
    speedLog(k) = state.vehicleSpeed;
end

figure;
plot(timeLog, speedLog * 3.6);
xlabel("Temps (s)"); ylabel("Vitesse vehicule (km/h)");
title("Test etape 1-2 : freinage en ligne droite depuis 80 km/h");
grid on;