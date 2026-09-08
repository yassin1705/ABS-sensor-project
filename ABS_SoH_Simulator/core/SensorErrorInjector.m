classdef SensorErrorInjector < handle
    % SENSORERRORINJECTOR  Applique des incertitudes nominales au signal
    % brut des capteurs avant sa reception par l'ECU.
    %
    % Cette V1 implemente uniquement des non-idealites compatibles avec un
    % capteur sain : latence fixe, jitter temporel absolu et jitter relatif
    % a la periode entre deux fronts. Le mode "ideal" laisse le signal
    % strictement inchange.

    properties (SetAccess = private)
        mode (1,1) string
        randomSeed (1,1) double
        latencySeconds (1,1) double
        absoluteJitterSeconds (1,1) double
        relativeJitterFraction (1,1) double
        faults struct
    end

    properties (Access = private)
        wheelState struct
        randomStream
    end

    properties (Constant, Access = private)
        WHEELS = ["FL", "FR", "RL", "RR"]
    end

    methods
        function obj = SensorErrorInjector(errorConfig, faults)
            arguments
                errorConfig (1,1) struct
                faults struct = struct([])
            end

            requiredFields = ["mode", "randomSeed", "latencyMs", ...
                "absoluteJitterUs", "relativeJitterPercent"];
            for field = requiredFields
                assert(isfield(errorConfig, field), ...
                    "Configuration d'incertitude manquante : %s.", field);
            end

            obj.mode = string(errorConfig.mode);
            assert(ismember(obj.mode, ["ideal", "healthy_realistic"]), ...
                "Le mode doit etre 'ideal' ou 'healthy_realistic'.");
            assert(errorConfig.randomSeed >= 0 && ...
                errorConfig.randomSeed == floor(errorConfig.randomSeed), ...
                "randomSeed doit etre un entier positif ou nul.");
            assert(errorConfig.latencyMs >= 0, "latencyMs doit etre positif ou nul.");
            assert(errorConfig.absoluteJitterUs >= 0, ...
                "absoluteJitterUs doit etre positif ou nul.");
            assert(errorConfig.relativeJitterPercent >= 0, ...
                "relativeJitterPercent doit etre positif ou nul.");

            obj.randomSeed = errorConfig.randomSeed;
            obj.latencySeconds = errorConfig.latencyMs * 1e-3;
            obj.absoluteJitterSeconds = errorConfig.absoluteJitterUs * 1e-6;
            obj.relativeJitterFraction = errorConfig.relativeJitterPercent / 100;
            obj.faults = faults;
            obj.validateFaults();
            obj.reset();
        end

        function reset(obj)
            obj.randomStream = RandStream("mt19937ar", "Seed", obj.randomSeed);
            obj.wheelState = struct();
            for wheel = obj.WHEELS
                obj.wheelState.(wheel) = struct( ...
                    "lastCleanEdgeTime", NaN, ...
                    "pendingEdgeTimes", zeros(1,0), ...
                    "outputLevel", false, ...
                    "totalOutputEdges", 0, ...
                    "totalDroppedEdges", 0);
            end
        end

        function alteredSignals = step(obj, cleanSignals, vehicleState)
            arguments
                obj
                cleanSignals (1,1) struct
                vehicleState (1,1) struct
            end

            assert(isfield(cleanSignals, "time"), ...
                "SensorSignals doit contenir le champ time.");
            assert(isfield(vehicleState, "wheelOmega"), ...
                "VehicleState doit contenir wheelOmega.");

            alteredSignals = struct("time", cleanSignals.time);
            for index = 1:numel(obj.WHEELS)
                wheel = obj.WHEELS(index);
                raw = cleanSignals.(wheel);
                state = obj.wheelState.(wheel);
                newEdges = reshape(raw.edgeTimes, 1, []);
                droppedEdgesThisStep = 0;

                for cleanEdgeTime = newEdges
                    if obj.shouldDropEdge(wheel, cleanEdgeTime)
                        droppedEdgesThisStep = droppedEdgesThisStep + 1;
                        state.totalDroppedEdges = state.totalDroppedEdges + 1;
                        state.lastCleanEdgeTime = cleanEdgeTime;
                        continue;
                    end

                    period = cleanEdgeTime - state.lastCleanEdgeTime;
                    if isnan(period) || period <= 0
                        omega = abs(vehicleState.wheelOmega(index));
                        if omega > eps
                            anglePerEdge = 2*pi / raw.edgesPerRevolution;
                            period = anglePerEdge / omega;
                        else
                            period = 0;
                        end
                    end

                    if obj.mode == "healthy_realistic"
                        relativeSigma = obj.relativeJitterFraction * period;
                        jitterSigma = hypot(obj.absoluteJitterSeconds, relativeSigma);
                        variableDelay = jitterSigma * randn(obj.randomStream);

                        % Le signal reste causal : un front ne peut pas sortir
                        % avant le franchissement magnetique qui l'a genere.
                        totalDelay = max(0, obj.latencySeconds + variableDelay);
                    else
                        totalDelay = 0;
                    end
                    state.pendingEdgeTimes(end + 1) = cleanEdgeTime + totalDelay;
                    state.lastCleanEdgeTime = cleanEdgeTime;
                end

                state.pendingEdgeTimes = sort(state.pendingEdgeTimes);
                due = state.pendingEdgeTimes <= cleanSignals.time;
                outputEdges = state.pendingEdgeTimes(due);
                state.pendingEdgeTimes = state.pendingEdgeTimes(~due);

                outputLevels = false(1, numel(outputEdges));
                for edgeIndex = 1:numel(outputEdges)
                    state.outputLevel = ~state.outputLevel;
                    outputLevels(edgeIndex) = state.outputLevel;
                end
                state.totalOutputEdges = state.totalOutputEdges + numel(outputEdges);
                [faultActive, faultType, faultSeverity] = ...
                    obj.activeFaultSummary(wheel, cleanSignals.time);

                alteredSignals.(wheel) = struct( ...
                    "wheel", wheel, ...
                    "time", cleanSignals.time, ...
                    "level", state.outputLevel, ...
                    "edgeTimes", outputEdges, ...
                    "edgeLevels", outputLevels, ...
                    "edgesThisStep", numel(outputEdges), ...
                    "totalEdges", state.totalOutputEdges, ...
                    "droppedEdgesThisStep", droppedEdgesThisStep, ...
                    "totalDroppedEdges", state.totalDroppedEdges, ...
                    "edgesPerRevolution", raw.edgesPerRevolution, ...
                    "healthy", ~faultActive, ...
                    "nominalErrorsApplied", obj.mode == "healthy_realistic", ...
                    "faultActive", faultActive, ...
                    "faultType", faultType, ...
                    "faultSeverity", faultSeverity);

                obj.wheelState.(wheel) = state;
            end
        end
    end

    methods (Access = private)
        function validateFaults(obj)
            supportedTypes = "intermittent_loss";
            validWheels = obj.WHEELS;
            for index = 1:numel(obj.faults)
                fault = obj.faults(index);
                required = ["wheel", "type", "startTime", "endTime", ...
                    "severity", "params"];
                for field = required
                    assert(isfield(fault, field), ...
                        "Champ de defaut manquant : %s.", field);
                end
                assert(ismember(string(fault.wheel), validWheels), ...
                    "Roue de defaut invalide : %s.", string(fault.wheel));
                assert(ismember(string(fault.type), supportedTypes), ...
                    "Type de defaut non supporte : %s.", string(fault.type));
                assert(fault.startTime >= 0 && fault.endTime > fault.startTime, ...
                    "Intervalle de defaut invalide.");
                assert(fault.severity >= 0 && fault.severity <= 1, ...
                    "La severite doit etre comprise entre 0 et 1.");
                assert(isfield(fault.params, "dropoutProbability"), ...
                    "intermittent_loss exige params.dropoutProbability.");
                assert(fault.params.dropoutProbability >= 0 && ...
                    fault.params.dropoutProbability <= 1, ...
                    "dropoutProbability doit etre comprise entre 0 et 1.");
            end
        end

        function drop = shouldDropEdge(obj, wheel, edgeTime)
            drop = false;
            for index = 1:numel(obj.faults)
                fault = obj.faults(index);
                active = string(fault.wheel) == wheel && ...
                    edgeTime >= fault.startTime && edgeTime <= fault.endTime;
                if ~active
                    continue;
                end

                switch string(fault.type)
                    case "intermittent_loss"
                        probability = min(1, ...
                            fault.severity * fault.params.dropoutProbability);
                        if rand(obj.randomStream) < probability
                            drop = true;
                            return;
                        end
                end
            end
        end

        function [active, type, severity] = activeFaultSummary(obj, wheel, time)
            active = false;
            type = "none";
            severity = 0;
            for index = 1:numel(obj.faults)
                fault = obj.faults(index);
                if string(fault.wheel) == wheel && ...
                        time >= fault.startTime && time <= fault.endTime
                    active = true;
                    type = string(fault.type);
                    severity = max(severity, fault.severity);
                end
            end
        end
    end
end
