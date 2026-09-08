function [outputPath, results] = run_simulation(configPath, durationSeconds, outputFolder)
% RUN_SIMULATION  Execute une simulation batch et exporte les resultats CSV.
%
% Usage minimal :
%   outputPath = run_simulation();
%
% Avec parametres :
%   outputPath = run_simulation("scenarios/scenario_reference.json", 20);
%
% Les defauts sont lus dans le champ "faults" du fichier JSON. La duree
% fournie en second argument remplace simulation.duration pour cette
% execution uniquement.

    arguments
        configPath (1,1) string = ""
        durationSeconds (1,1) double = NaN
        outputFolder (1,1) string = ""
    end

    projectRoot = fileparts(mfilename("fullpath"));
    addpath(fullfile(projectRoot, "core"));

    if strlength(configPath) == 0
        configPath = fullfile(projectRoot, "scenarios", "scenario_reference.json");
    elseif ~isfile(configPath)
        candidate = fullfile(projectRoot, configPath);
        if isfile(candidate)
            configPath = candidate;
        end
    end

    if strlength(outputFolder) == 0
        outputFolder = fullfile(projectRoot, "simulation_results");
    end

    config = SimulationConfig(configPath);
    if isnan(durationSeconds)
        durationSeconds = config.simulation.duration;
    end
    assert(durationSeconds > 0, "La duree de simulation doit etre positive.");

    dt = config.simulation.dt;
    numberOfSteps = floor(durationSeconds / dt);
    assert(numberOfSteps >= 1, ...
        "La duree doit etre au moins egale au pas de simulation.");

    controller = SimulationController(config);
    logger = SimulationDataLogger(numberOfSteps, config);

    for stepIndex = 1:numberOfSteps
        commandTime = (stepIndex - 1) * dt;
        frame = controller.stepAt(commandTime);
        logger.append(frame);
    end

    outputPath = logger.exportCsv(outputFolder);
    results = logger.toTable();

    fprintf("Simulation terminee : %d pas, %.3f s.\n", ...
        numberOfSteps, numberOfSteps * dt);
    fprintf("Resultats CSV : %s\n", outputPath);
end
