% TEST_STEP2_1_BATCH_SIMULATION
% Verifie l'execution batch, l'export CSV et l'application des defauts.

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(projectRoot);
addpath(fullfile(projectRoot, "core"));

configPath = fullfile(projectRoot, "scenarios", "scenario_reference.json");
testOutputFolder = fullfile(tempdir, "abs_soh_simulator_tests");
[outputPath, results] = run_simulation(configPath, 10.0, testOutputFolder);

assert(isfile(outputPath), "Le fichier CSV doit etre cree.");
assert(height(results) == 1000, "Le nombre de lignes CSV est incorrect.");
assert(ismember("wheel_speed_ecu_FL_mps", string(results.Properties.VariableNames)), ...
    "La vitesse ECU FL doit etre presente dans le CSV.");
assert(any(results.fault_active_FL), "Le defaut FL doit devenir actif.");
assert(sum(results.dropped_edges_FL) > 0, ...
    "Le defaut intermittent doit supprimer des fronts FL.");
assert(~any(results.fault_active_FR), ...
    "Le defaut FL ne doit pas activer le label FR.");

fprintf("Test batch reussi : %d lignes, %d fronts FL supprimes.\n", ...
    height(results), sum(results.dropped_edges_FL));
