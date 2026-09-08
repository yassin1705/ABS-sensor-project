classdef ExperimentDatasetGenerator < handle
    % EXPERIMENTDATASETGENERATOR  Genere un CSV multi-simulations.
    %
    % Exemple :
    %   distribution = struct( ...
    %       "normal_rolling", 0.30, ...
    %       "wheel_lock",     0.30, ...
    %       "sliding",        0.30, ...
    %       "random",         0.10);
    %   generator = ExperimentDatasetGenerator();
    %   [csvPath, data, manifest] = generator.generate( ...
    %       distribution, 100, 5.0);
    %
    % Toutes les simulations utilisent la meme duree et le meme dt.
    % Les defauts de mesure anormaux sont desactives : ce generateur est
    % destine au premier modele predictif de dynamique saine.

    properties (SetAccess = private)
        baseConfigPath (1,1) string
        outputFolder (1,1) string
        randomSeed (1,1) double
    end

    properties (Constant, Access = private)
        PHENOMENA = ["normal_rolling", "wheel_lock", "sliding", "random"]
    end

    methods
        function obj = ExperimentDatasetGenerator(baseConfigPath, ...
                outputFolder, randomSeed)
            arguments
                baseConfigPath (1,1) string = ""
                outputFolder (1,1) string = ""
                randomSeed (1,1) double {mustBeNonnegative, mustBeInteger} = 1000
            end

            projectRoot = fileparts(fileparts(mfilename("fullpath")));
            if strlength(baseConfigPath) == 0
                baseConfigPath = fullfile(projectRoot, "scenarios", ...
                    "scenario_physics_dataset.json");
            end
            if strlength(outputFolder) == 0
                outputFolder = fullfile(projectRoot, "simulation_results");
            end
            assert(isfile(baseConfigPath), ...
                "Configuration physique de base introuvable : %s", ...
                baseConfigPath);

            obj.baseConfigPath = baseConfigPath;
            obj.outputFolder = outputFolder;
            obj.randomSeed = randomSeed;
        end

        function [outputPath, dataset, manifest] = generate(obj, ...
                distribution, numberOfSimulations, durationSeconds)
            arguments
                obj
                distribution
                numberOfSimulations (1,1) double ...
                    {mustBePositive, mustBeInteger}
                durationSeconds (1,1) double {mustBePositive}
            end

            assert(durationSeconds >= 2.0, ...
                "La duree fixe doit etre au moins egale a 2 secondes.");
            probabilities = obj.parseDistribution(distribution);
            counts = obj.allocateCounts(probabilities, numberOfSimulations);
            phenomena = obj.expandAndShufflePhenomena(counts);

            template = jsondecode(fileread(obj.baseConfigPath));
            dt = template.simulation.dt;
            numberOfSteps = floor(durationSeconds / dt);
            assert(numberOfSteps >= 1, ...
                "La duree doit etre au moins egale au pas de simulation.");
            assert(abs(numberOfSteps * dt - durationSeconds) < 1e-9, ...
                "La duree doit etre un multiple exact du pas de simulation.");

            dataset = table();
            manifestRows = repmat(obj.emptyManifestRow(), ...
                numberOfSimulations, 1);
            stream = RandStream("mt19937ar", "Seed", obj.randomSeed);

            for simulationIndex = 1:numberOfSimulations
                phenomenon = phenomena(simulationIndex);
                concreteSeed = obj.randomSeed + simulationIndex - 1;
                [rawConfig, generated] = obj.buildConcreteConfig( ...
                    template, phenomenon, durationSeconds, ...
                    concreteSeed, stream);

                configPath = obj.writeTemporaryConfig(rawConfig);
                cleanup = onCleanup(@()obj.deleteTemporaryConfig(configPath));
                config = SimulationConfig(configPath);
                controller = SimulationController(config);
                logger = SimulationDataLogger(numberOfSteps, config);

                for stepIndex = 1:numberOfSteps
                    commandTime = (stepIndex - 1) * dt;
                    logger.append(controller.stepAt(commandTime));
                end

                result = logger.toTable();
                n = height(result);
                result = addvars(result, ...
                    repmat(simulationIndex, n, 1), ...
                    repmat(phenomenon, n, 1), ...
                    repmat(string(generated.driverProfile), n, 1), ...
                    repmat(concreteSeed, n, 1), ...
                    repmat(generated.initialSpeedKmh, n, 1), ...
                    repmat(generated.muRoad, n, 1), ...
                    'Before', 1, ...
                    'NewVariableNames', { ...
                    'simulation_id', 'requested_phenomenon', ...
                    'driver_profile', 'simulation_seed', ...
                    'initial_speed_kmh', 'generated_mu_road'});

                if simulationIndex == 1
                    dataset = result;
                else
                    dataset = [dataset; result]; %#ok<AGROW>
                end

                manifestRows(simulationIndex) = obj.buildManifestRow( ...
                    simulationIndex, phenomenon, generated, result);
                clear cleanup

                fprintf("Simulation %d/%d : %s (%d pas).\n", ...
                    simulationIndex, numberOfSimulations, phenomenon, n);
            end

            manifest = struct2table(manifestRows);
            if ~isfolder(obj.outputFolder)
                mkdir(obj.outputFolder);
            end
            timestamp = string(datetime("now", ...
                "Format", "yyyyMMdd_HHmmss_SSS"));
            outputPath = fullfile(obj.outputFolder, ...
                "abs_physics_dataset_" + timestamp + ".csv");
            writetable(dataset, outputPath);
            fprintf("Dataset termine : %d simulations, %d lignes.\n", ...
                numberOfSimulations, height(dataset));
            fprintf("CSV : %s\n", outputPath);
        end
    end

    methods (Access = private)
        function probabilities = parseDistribution(obj, distribution)
            if isnumeric(distribution)
                assert(isvector(distribution) && numel(distribution) == 4, ...
                    "Une distribution numerique doit contenir 4 valeurs.");
                probabilities = reshape(double(distribution), 1, 4);
            elseif isstruct(distribution)
                probabilities = zeros(1,4);
                aliases = { ...
                    ["normal_rolling", "normal", "roulement_normal"], ...
                    ["wheel_lock", "lock", "blockage"], ...
                    ["sliding", "glissement"], ...
                    ["random", "aleatoire"]};
                for index = 1:4
                    found = false;
                    for alias = aliases{index}
                        if isfield(distribution, alias)
                            probabilities(index) = distribution.(alias);
                            found = true;
                            break;
                        end
                    end
                    assert(found, ...
                        "Probabilite manquante pour %s.", ...
                        obj.PHENOMENA(index));
                end
            else
                error("ExperimentDatasetGenerator:distributionType", ...
                    "La distribution doit etre un vecteur ou une struct.");
            end

            assert(all(isfinite(probabilities)) && all(probabilities >= 0), ...
                "Les probabilites doivent etre finies et positives ou nulles.");
            total = sum(probabilities);
            assert(total > 0, "La somme des probabilites doit etre positive.");
            probabilities = probabilities / total;
        end

        function counts = allocateCounts(~, probabilities, totalCount)
            exactCounts = probabilities * totalCount;
            counts = floor(exactCounts);
            remaining = totalCount - sum(counts);
            [~, order] = sort(exactCounts - counts, "descend");
            counts(order(1:remaining)) = counts(order(1:remaining)) + 1;
        end

        function phenomena = expandAndShufflePhenomena(obj, counts)
            phenomena = strings(1, sum(counts));
            cursor = 1;
            for index = 1:numel(obj.PHENOMENA)
                nextCursor = cursor + counts(index);
                phenomena(cursor:nextCursor-1) = obj.PHENOMENA(index);
                cursor = nextCursor;
            end
            stream = RandStream("mt19937ar", "Seed", obj.randomSeed);
            phenomena = phenomena(randperm(stream, numel(phenomena)));
        end

        function [raw, generated] = buildConcreteConfig(obj, template, ...
                phenomenon, duration, seed, stream)
            raw = template;
            raw.meta.name = sprintf("generated_%s_%06d", phenomenon, seed);
            raw.meta.description = sprintf( ...
                "Experience physique generee : %s", phenomenon);
            raw.meta.seed = seed;
            raw.meta.requestedPhenomenon = char(phenomenon);
            raw.simulation.duration = duration;
            raw.sensorErrors.randomSeed = seed;
            if isfield(raw, "faults")
                raw = rmfield(raw, "faults");
            end

            initialSpeedKmh = obj.uniform(stream, 35, 95);
            muRoad = obj.uniform(stream, 0.7, 1.0);
            driverProfile = "normal_straight";
            throttle = 0;
            brake = 0;
            steering = 0;
            actionStart = obj.uniform(stream, 0.20*duration, 0.35*duration);
            actionEnd = min(0.75*duration, actionStart + 0.30*duration);

            switch phenomenon
                case "normal_rolling"
                    variant = randi(stream, 4);
                    muRoad = obj.uniform(stream, 0.65, 1.0);
                    throttle = obj.uniform(stream, 0.05, 0.35);
                    actionStart = 0;
                    actionEnd = duration;
                    switch variant
                        case 1
                            driverProfile = "normal_straight";
                        case 2
                            driverProfile = "turn_left";
                            steering = -deg2rad(obj.uniform(stream, 3, 12));
                        case 3
                            driverProfile = "turn_right";
                            steering = deg2rad(obj.uniform(stream, 3, 12));
                        otherwise
                            driverProfile = "acceleration";
                            initialSpeedKmh = obj.uniform(stream, 5, 45);
                            throttle = obj.uniform(stream, 0.45, 0.85);
                    end

                case "wheel_lock"
                    driverProfile = "long_hard_braking";
                    initialSpeedKmh = obj.uniform(stream, 55, 100);
                    muRoad = obj.uniform(stream, 0.06, 0.18);
                    brake = obj.uniform(stream, 0.85, 1.0);
                    actionEnd = min(0.82*duration, ...
                        actionStart + obj.uniform(stream, ...
                        0.28*duration, 0.45*duration));

                case "sliding"
                    driverProfile = "short_hard_braking";
                    initialSpeedKmh = obj.uniform(stream, 50, 100);
                    muRoad = obj.uniform(stream, 0.30, 0.55);
                    brake = obj.uniform(stream, 0.12, 0.25);
                    actionEnd = min(0.72*duration, ...
                        actionStart + obj.uniform(stream, 0.15, 0.35));

                case "random"
                    variant = randi(stream, 5);
                    muRoad = obj.uniform(stream, 0.25, 1.0);
                    switch variant
                        case 1
                            driverProfile = "random_acceleration";
                            initialSpeedKmh = obj.uniform(stream, 0, 60);
                            throttle = obj.uniform(stream, 0.25, 1.0);
                        case 2
                            driverProfile = "random_braking";
                            brake = obj.uniform(stream, 0.15, 0.75);
                        case 3
                            driverProfile = "random_turn_left";
                            throttle = obj.uniform(stream, 0, 0.4);
                            steering = -deg2rad(obj.uniform(stream, 2, 18));
                        case 4
                            driverProfile = "random_turn_right";
                            throttle = obj.uniform(stream, 0, 0.4);
                            steering = deg2rad(obj.uniform(stream, 2, 18));
                        otherwise
                            driverProfile = "random_braking_turn";
                            brake = obj.uniform(stream, 0.15, 0.65);
                            steeringSign = 2 * (rand(stream) >= 0.5) - 1;
                            steering = steeringSign * ...
                                deg2rad(obj.uniform(stream, 2, 12));
                    end
            end

            raw.vehicle.muRoad = muRoad;
            raw.road.muRoad = muRoad;
            raw.scenario.type = "generated_segments";
            raw.scenario.initialSpeed_kmh = initialSpeedKmh;
            raw.scenario.defaultCommand = struct( ...
                "throttle", 0, "brake", 0, "steering", 0);
            raw.scenario.segments = obj.makeSegments(duration, ...
                actionStart, actionEnd, throttle, brake, steering);

            raw.meta.driverProfile = char(driverProfile);
            raw.meta.generatedMuRoad = muRoad;
            generated = struct( ...
                "driverProfile", driverProfile, ...
                "initialSpeedKmh", initialSpeedKmh, ...
                "muRoad", muRoad, ...
                "actionStart", actionStart, ...
                "actionEnd", actionEnd, ...
                "throttle", throttle, ...
                "brake", brake, ...
                "steering", steering);
        end

        function segments = makeSegments(~, duration, actionStart, ...
                actionEnd, throttle, brake, steering)
            segments = struct("startTime", {}, "endTime", {}, ...
                "throttle", {}, "brake", {}, "steering", {});
            if actionStart > 0
                segments(end + 1) = struct( ...
                    "startTime", 0, "endTime", actionStart, ...
                    "throttle", 0, "brake", 0, "steering", 0);
            end
            segments(end + 1) = struct( ...
                "startTime", actionStart, "endTime", actionEnd, ...
                "throttle", throttle, "brake", brake, ...
                "steering", steering);
            if actionEnd < duration
                segments(end + 1) = struct( ...
                    "startTime", actionEnd, "endTime", duration, ...
                    "throttle", 0, "brake", 0, "steering", 0);
            end
        end

        function configPath = writeTemporaryConfig(~, rawConfig)
            configPath = string(tempname) + ".json";
            fileId = fopen(configPath, "w");
            assert(fileId >= 0, ...
                "Impossible de creer la configuration temporaire.");
            closeFile = onCleanup(@()fclose(fileId));
            fwrite(fileId, jsonencode(rawConfig, PrettyPrint=true), "char");
            clear closeFile
        end

        function deleteTemporaryConfig(~, configPath)
            if isfile(configPath)
                delete(configPath);
            end
        end

        function value = uniform(~, stream, lower, upper)
            value = lower + (upper - lower) * rand(stream);
        end

        function row = emptyManifestRow(~)
            row = struct( ...
                "simulation_id", 0, ...
                "requested_phenomenon", "", ...
                "driver_profile", "", ...
                "seed", 0, ...
                "initial_speed_kmh", 0, ...
                "mu_road", 0, ...
                "action_start_s", 0, ...
                "action_end_s", 0, ...
                "maximum_abs_slip", 0, ...
                "any_wheel_locked", false);
        end

        function row = buildManifestRow(~, simulationIndex, phenomenon, ...
                generated, result)
            slipNames = "wheel_slip_ratio_" + ["FL", "FR", "RL", "RR"];
            lockNames = "wheel_locked_" + ["FL", "FR", "RL", "RR"];
            slip = result{:, cellstr(slipNames)};
            locked = result{:, cellstr(lockNames)};
            row = struct( ...
                "simulation_id", simulationIndex, ...
                "requested_phenomenon", phenomenon, ...
                "driver_profile", string(generated.driverProfile), ...
                "seed", 0, ...
                "initial_speed_kmh", generated.initialSpeedKmh, ...
                "mu_road", generated.muRoad, ...
                "action_start_s", generated.actionStart, ...
                "action_end_s", generated.actionEnd, ...
                "maximum_abs_slip", max(abs(slip), [], "all"), ...
                "any_wheel_locked", any(locked, "all"));
            row.seed = result.simulation_seed(1);
        end
    end
end
