classdef ABSECU < handle
    % ABSECU  Reconstruction nominale des vitesses de roue a partir des
    % fronts emis par les capteurs ABS actifs.
    %
    % Les fronts sont captures des qu'ils sont recus. L'ECU publie ensuite
    % une mesure a sampleTime fixe en utilisant la periode entre les deux
    % derniers fronts. Aucun filtre ni defaut n'est modele dans cette V1.

    properties (SetAccess = private)
        sampleTime (1,1) double
        noPulseTimeout (1,1) double
        wheelRadius (1,1) double
        edgesPerRevolution struct
    end

    properties (Access = private)
        wheelState struct
        nextSampleTime (1,1) double = 0
        lastInputTime (1,1) double = 0
        lastOutput struct
    end

    properties (Constant, Access = private)
        WHEELS = ["FL", "FR", "RL", "RR"]
        TIME_TOLERANCE = 1e-12
    end

    methods
        function obj = ABSECU(ecuConfig, sensorConfig, wheelRadius)
            arguments
                ecuConfig (1,1) struct
                sensorConfig (1,1) struct
                wheelRadius (1,1) double {mustBePositive}
            end
            assert(isfield(ecuConfig, "sampleTime") && ecuConfig.sampleTime > 0, ...
                "La configuration ECU doit contenir un sampleTime positif.");
            assert(isfield(ecuConfig, "noPulseTimeout") && ecuConfig.noPulseTimeout > 0, ...
                "La configuration ECU doit contenir un noPulseTimeout positif.");
            assert(isfield(sensorConfig, "edgesPerRevolution"), ...
                "La configuration capteur doit contenir edgesPerRevolution.");

            obj.sampleTime = ecuConfig.sampleTime;
            obj.noPulseTimeout = ecuConfig.noPulseTimeout;
            obj.wheelRadius = wheelRadius;
            obj.edgesPerRevolution = sensorConfig.edgesPerRevolution;
            obj.reset();
        end

        function reset(obj)
            obj.wheelState = struct();
            for wheel = obj.WHEELS
                obj.wheelState.(wheel) = struct( ...
                    "pendingEdgeTimes", zeros(1,0), ...
                    "previousEdgeTime", NaN, ...
                    "lastEdgeTime", NaN, ...
                    "totalCapturedEdges", 0, ...
                    "capturedAtLastSample", 0);
            end
            obj.nextSampleTime = obj.sampleTime;
            obj.lastInputTime = 0;
            obj.lastOutput = obj.emptyMeasurement(0);
        end

        function measurements = step(obj, sensorSignals)
            % STEP Capture les fronts recus, puis execute tous les ticks
            % ECU echus jusqu'a sensorSignals.time.
            arguments
                obj
                sensorSignals (1,1) struct
            end
            assert(isfield(sensorSignals, "time"), ...
                "Les signaux capteurs doivent contenir un champ time.");

            time = sensorSignals.time;
            if time < obj.lastInputTime - obj.TIME_TOLERANCE
                error("ABSECU:timeRegression", "Le temps des signaux doit etre croissant.");
            end

            obj.enqueueEdges(sensorSignals);

            % Les fronts sont captures dans leur ordre temporel. Cette boucle
            % reproduit correctement plusieurs ticks ECU meme si l'appelant
            % utilise un pas de simulation plus grand que sampleTime.
            while obj.nextSampleTime <= time + obj.TIME_TOLERANCE
                obj.lastOutput = obj.sampleAt(obj.nextSampleTime);
                obj.nextSampleTime = obj.nextSampleTime + obj.sampleTime;
            end

            obj.lastInputTime = time;
            measurements = obj.lastOutput;
        end
    end

    methods (Access = private)
        function enqueueEdges(obj, sensorSignals)
            for wheel = obj.WHEELS
                assert(isfield(sensorSignals, wheel), ...
                    "Signal manquant pour la roue %s.", wheel);
                raw = sensorSignals.(wheel);
                assert(isfield(raw, "edgeTimes"), ...
                    "Le signal de la roue %s doit contenir edgeTimes.", wheel);

                state = obj.wheelState.(wheel);
                newEdges = reshape(raw.edgeTimes, 1, []);
                if ~isempty(newEdges)
                    assert(all(diff(newEdges) >= 0), ...
                        "Les fronts de la roue %s doivent etre tries.", wheel);
                    state.pendingEdgeTimes = [state.pendingEdgeTimes, newEdges];
                end
                obj.wheelState.(wheel) = state;
            end
        end

        function measurements = sampleAt(obj, sampleTime)
            speed = NaN(1,4);
            valid = false(1,4);
            status = strings(1,4);
            edgePeriod = NaN(1,4);
            edgeAge = NaN(1,4);

            for index = 1:numel(obj.WHEELS)
                wheel = obj.WHEELS(index);
                state = obj.captureUntil(wheel, sampleTime);

                if isnan(state.lastEdgeTime)
                    status(index) = "initializing";
                else
                    edgeAge(index) = sampleTime - state.lastEdgeTime;

                    if edgeAge(index) > obj.noPulseTimeout
                        speed(index) = 0;
                        valid(index) = true;
                        status(index) = "zero_timeout";
                    elseif ~isnan(state.previousEdgeTime)
                        edgePeriod(index) = state.lastEdgeTime - state.previousEdgeTime;
                        anglePerEdge = 2*pi / obj.edgesPerRevolution.(wheel);
                        speed(index) = obj.wheelRadius * anglePerEdge / edgePeriod(index);
                        valid(index) = true;

                        if state.totalCapturedEdges > state.capturedAtLastSample
                            status(index) = "period";
                        else
                            status(index) = "held";
                        end
                    else
                        status(index) = "initializing";
                    end
                end

                state.capturedAtLastSample = state.totalCapturedEdges;
                obj.wheelState.(wheel) = state;
            end

            measurements = struct( ...
                "time", sampleTime, ...
                "wheelSpeedMeasured", speed, ...
                "valid", valid, ...
                "status", status, ...
                "edgePeriod", edgePeriod, ...
                "ageSinceLastEdge", edgeAge);
        end

        function state = captureUntil(obj, wheel, sampleTime)
            state = obj.wheelState.(wheel);
            due = state.pendingEdgeTimes <= sampleTime + obj.TIME_TOLERANCE;
            dueTimes = state.pendingEdgeTimes(due);
            state.pendingEdgeTimes = state.pendingEdgeTimes(~due);

            for edgeTime = dueTimes
                state.previousEdgeTime = state.lastEdgeTime;
                state.lastEdgeTime = edgeTime;
                state.totalCapturedEdges = state.totalCapturedEdges + 1;
            end
            obj.wheelState.(wheel) = state;
        end

        function measurements = emptyMeasurement(~, time)
            measurements = struct( ...
                "time", time, ...
                "wheelSpeedMeasured", NaN(1,4), ...
                "valid", false(1,4), ...
                "status", repmat("initializing", 1, 4), ...
                "edgePeriod", NaN(1,4), ...
                "ageSinceLastEdge", NaN(1,4));
        end
    end
end
