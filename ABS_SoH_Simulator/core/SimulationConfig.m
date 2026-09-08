classdef SimulationConfig < handle
    % SIMULATIONCONFIG  Charge, valide et expose les parametres d'une experience.
    %
    % Usage:
    %   cfg = SimulationConfig("scenarios/scenario_reference.json");
    %   disp(cfg.vehicle.mass)
    %   rng(cfg.meta.seed);   % a appeler en debut de SimulationController

    properties (SetAccess = private)
        meta        struct
        simulation  struct
        vehicle     struct
        dynamics    struct
        brakeSystem struct
        road        struct
        scenario    struct
        sensor      struct
        sensorErrors struct
        ecu         struct
        faults      struct  % array de structs, un par defaut
        sourceFile  string
    end

    methods
        function obj = SimulationConfig(jsonPath)
            arguments
                jsonPath (1,1) string = "scenarios/scenario_reference.json"
            end
            if ~isfile(jsonPath)
                error("SimulationConfig:fileNotFound", ...
                    "Fichier de configuration introuvable : %s", jsonPath);
            end

            raw = jsondecode(fileread(jsonPath));
            obj.sourceFile = jsonPath;

            obj.meta       = raw.meta;
            obj.simulation = raw.simulation;
            obj.vehicle    = raw.vehicle;
            obj.dynamics   = obj.fieldOrDefault(raw, "dynamics", ...
                struct("model", "legacy"));
            obj.brakeSystem = obj.fieldOrDefault(raw, "brakeSystem", struct());
            obj.road       = obj.fieldOrDefault(raw, "road", ...
                struct("muRoad", raw.vehicle.muRoad));
            obj.scenario   = raw.scenario;
            obj.sensor     = raw.sensor;
            obj.sensorErrors = raw.sensorErrors;
            obj.ecu        = raw.ecu;

            if isfield(raw, "faults")
                obj.faults = raw.faults;
            else
                obj.faults = struct([]);
            end

            obj.validate();
        end

        function validate(obj)
            % Verifications minimales de coherence physique et logique.
            assert(obj.simulation.dt > 0, "dt doit etre positif.");
            assert(obj.simulation.duration > obj.simulation.dt, ...
                "duration doit etre superieure a dt.");
            assert(obj.vehicle.mass > 0, "La masse du vehicule doit etre positive.");
            assert(obj.vehicle.wheelRadius > 0, "Le rayon de roue doit etre positif.");
            assert(obj.vehicle.track > 0, "La voie doit etre positive.");
            assert(obj.vehicle.wheelbase > 0, "L'empattement doit etre positif.");

            assert(isfield(obj.dynamics, "model"), ...
                "La configuration dynamics doit contenir model.");
            assert(ismember(string(obj.dynamics.model), ...
                ["legacy", "longitudinal_wheel_slip"]), ...
                "dynamics.model doit etre legacy ou longitudinal_wheel_slip.");

            if string(obj.dynamics.model) == "longitudinal_wheel_slip"
                requiredDynamics = ["wheelInertia", "maxDriveTorque", ...
                    "slipShapeFactor", "rollingResistanceCoefficient", ...
                    "aerodynamicDragCoefficient", "cgHeight", ...
                    "distanceCGToFrontAxle", "lowSpeedRegularization", ...
                    "lockSlipThreshold", "minimumLockSpeed"];
                for field = requiredDynamics
                    assert(isfield(obj.dynamics, field), ...
                        "Configuration dynamics manquante : %s.", field);
                end
                assert(obj.dynamics.wheelInertia > 0, ...
                    "dynamics.wheelInertia doit etre positif.");
                assert(obj.dynamics.maxDriveTorque >= 0, ...
                    "dynamics.maxDriveTorque doit etre positif ou nul.");
                assert(obj.dynamics.slipShapeFactor > 0, ...
                    "dynamics.slipShapeFactor doit etre positif.");
                assert(obj.dynamics.cgHeight >= 0, ...
                    "dynamics.cgHeight doit etre positif ou nul.");
                assert(obj.dynamics.distanceCGToFrontAxle > 0 && ...
                    obj.dynamics.distanceCGToFrontAxle < obj.vehicle.wheelbase, ...
                    "distanceCGToFrontAxle doit etre dans l'empattement.");
                assert(obj.dynamics.lowSpeedRegularization > 0, ...
                    "lowSpeedRegularization doit etre positif.");
                assert(obj.dynamics.lockSlipThreshold > 0 && ...
                    obj.dynamics.lockSlipThreshold <= 1, ...
                    "lockSlipThreshold doit etre dans ]0,1].");
                assert(obj.dynamics.minimumLockSpeed >= 0, ...
                    "minimumLockSpeed doit etre positif ou nul.");

                requiredBrake = ["maxBrakeTorque", "frontDistribution"];
                for field = requiredBrake
                    assert(isfield(obj.brakeSystem, field), ...
                        "Configuration brakeSystem manquante : %s.", field);
                end
                assert(obj.brakeSystem.maxBrakeTorque > 0, ...
                    "brakeSystem.maxBrakeTorque doit etre positif.");
                assert(obj.brakeSystem.frontDistribution >= 0 && ...
                    obj.brakeSystem.frontDistribution <= 1, ...
                    "frontDistribution doit etre entre 0 et 1.");

                assert(isfield(obj.road, "muRoad") && obj.road.muRoad > 0, ...
                    "road.muRoad doit etre positif.");
            end

            validWheels = ["FL", "FR", "RL", "RR"];
            assert(isfield(obj.sensor, "edgesPerRevolution"), ...
                "La configuration capteur doit contenir edgesPerRevolution.");
            for wheel = validWheels
                assert(isfield(obj.sensor.edgesPerRevolution, wheel), ...
                    "Nombre de fronts manquant pour la roue %s.", wheel);
                nEdges = obj.sensor.edgesPerRevolution.(wheel);
                assert(isscalar(nEdges) && nEdges > 0 && nEdges == floor(nEdges), ...
                    "Le nombre de fronts doit etre un entier positif pour la roue %s.", wheel);
            end

            requiredErrorFields = ["mode", "randomSeed", "latencyMs", ...
                "absoluteJitterUs", "relativeJitterPercent"];
            for field = requiredErrorFields
                assert(isfield(obj.sensorErrors, field), ...
                    "Configuration sensorErrors manquante : %s.", field);
            end
            assert(ismember(string(obj.sensorErrors.mode), ...
                ["ideal", "healthy_realistic"]), ...
                "sensorErrors.mode doit etre ideal ou healthy_realistic.");
            assert(obj.sensorErrors.latencyMs >= 0, ...
                "sensorErrors.latencyMs doit etre positif ou nul.");
            assert(obj.sensorErrors.absoluteJitterUs >= 0, ...
                "sensorErrors.absoluteJitterUs doit etre positif ou nul.");
            assert(obj.sensorErrors.relativeJitterPercent >= 0, ...
                "sensorErrors.relativeJitterPercent doit etre positif ou nul.");

            assert(isfield(obj.ecu, "sampleTime") && obj.ecu.sampleTime > 0, ...
                "La configuration ECU doit contenir sampleTime > 0.");
            assert(isfield(obj.ecu, "noPulseTimeout") && obj.ecu.noPulseTimeout > 0, ...
                "La configuration ECU doit contenir noPulseTimeout > 0.");

            for k = 1:numel(obj.faults)
                assert(ismember(obj.faults(k).wheel, validWheels), ...
                    "Roue de defaut invalide : %s", obj.faults(k).wheel);
                assert(isfield(obj.faults(k), "type"), ...
                    "Le defaut %d doit contenir un type.", k);
                assert(isfield(obj.faults(k), "startTime") && ...
                    isfield(obj.faults(k), "endTime") && ...
                    obj.faults(k).startTime >= 0 && ...
                    obj.faults(k).endTime > obj.faults(k).startTime, ...
                    "Intervalle invalide pour le defaut %d.", k);
                assert(isfield(obj.faults(k), "severity") && ...
                    obj.faults(k).severity >= 0 && obj.faults(k).severity <= 1, ...
                    "La severite du defaut %d doit etre entre 0 et 1.", k);
            end
        end

        function n = numSteps(obj)
            n = floor(obj.simulation.duration / obj.simulation.dt);
        end

        function seed = getSeed(obj)
            if isfield(obj.meta, "seed")
                seed = obj.meta.seed;
            else
                seed = 0;
            end
        end
    end

    methods (Static, Access = private)
        function value = fieldOrDefault(source, field, defaultValue)
            if isfield(source, field)
                value = source.(field);
            else
                value = defaultValue;
            end
        end
    end
end
