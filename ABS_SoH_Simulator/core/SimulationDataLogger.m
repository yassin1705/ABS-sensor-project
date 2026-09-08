classdef SimulationDataLogger < handle
    % SIMULATIONDATALOGGER  Accumule des sorties a taille fixe et les
    % exporte sous forme de table CSV exploitable pour l'analyse ou l'IA.

    properties (SetAccess = private)
        capacity (1,1) double
        count (1,1) double = 0
        config
    end

    properties (Access = private)
        time
        throttle
        brake
        steering
        vehicleSpeed
        yawRate
        accelerationX
        accelerationY
        wheelSpeedTrue
        wheelGroundSpeed
        wheelPeripheralSpeed
        wheelAngularAcceleration
        wheelSlipRatio
        normalLoad
        longitudinalTireForce
        brakeTorque
        roadFriction
        wheelLocked
        wheelSpeedECU
        signedError
        ecuValid
        ecuStatus
        signalLevel
        edgesThisStep
        droppedEdgesThisStep
        faultActive
        faultType
        faultSeverity
    end

    properties (Constant, Access = private)
        WHEELS = ["FL", "FR", "RL", "RR"]
    end

    methods
        function obj = SimulationDataLogger(capacity, config)
            arguments
                capacity (1,1) double {mustBePositive, mustBeInteger}
                config (1,1) SimulationConfig
            end
            obj.capacity = capacity;
            obj.config = config;

            obj.time = NaN(capacity,1);
            obj.throttle = NaN(capacity,1);
            obj.brake = NaN(capacity,1);
            obj.steering = NaN(capacity,1);
            obj.vehicleSpeed = NaN(capacity,1);
            obj.yawRate = NaN(capacity,1);
            obj.accelerationX = NaN(capacity,1);
            obj.accelerationY = NaN(capacity,1);
            obj.wheelSpeedTrue = NaN(capacity,4);
            obj.wheelGroundSpeed = NaN(capacity,4);
            obj.wheelPeripheralSpeed = NaN(capacity,4);
            obj.wheelAngularAcceleration = NaN(capacity,4);
            obj.wheelSlipRatio = NaN(capacity,4);
            obj.normalLoad = NaN(capacity,4);
            obj.longitudinalTireForce = NaN(capacity,4);
            obj.brakeTorque = NaN(capacity,4);
            obj.roadFriction = NaN(capacity,4);
            obj.wheelLocked = false(capacity,4);
            obj.wheelSpeedECU = NaN(capacity,4);
            obj.signedError = NaN(capacity,4);
            obj.ecuValid = false(capacity,4);
            obj.ecuStatus = strings(capacity,4);
            obj.signalLevel = false(capacity,4);
            obj.edgesThisStep = zeros(capacity,4);
            obj.droppedEdgesThisStep = zeros(capacity,4);
            obj.faultActive = false(capacity,4);
            obj.faultType = repmat("none", capacity,4);
            obj.faultSeverity = zeros(capacity,4);
        end

        function append(obj, frame)
            assert(obj.count < obj.capacity, ...
                "La capacite du logger est depassee.");
            obj.count = obj.count + 1;
            row = obj.count;

            command = frame.command;
            truth = frame.truth;
            measurements = frame.ecuMeasurements;

            obj.time(row) = truth.time;
            obj.throttle(row) = command.throttle;
            obj.brake(row) = command.brake;
            obj.steering(row) = command.steering;
            obj.vehicleSpeed(row) = truth.vehicleSpeed;
            obj.yawRate(row) = truth.yawRate;
            obj.accelerationX(row) = truth.accelerationX;
            obj.accelerationY(row) = truth.accelerationY;
            obj.wheelSpeedTrue(row,:) = truth.wheelSpeedTrue;
            obj.wheelGroundSpeed(row,:) = obj.fieldOrDefault( ...
                truth, "wheelGroundSpeed", truth.wheelSpeedTrue);
            obj.wheelPeripheralSpeed(row,:) = obj.fieldOrDefault( ...
                truth, "wheelPeripheralSpeed", truth.wheelSpeedTrue);
            obj.wheelAngularAcceleration(row,:) = obj.fieldOrDefault( ...
                truth, "wheelAngularAcceleration", NaN(1,4));
            obj.wheelSlipRatio(row,:) = obj.fieldOrDefault( ...
                truth, "wheelSlipRatio", zeros(1,4));
            obj.normalLoad(row,:) = obj.fieldOrDefault( ...
                truth, "normalLoad", NaN(1,4));
            obj.longitudinalTireForce(row,:) = obj.fieldOrDefault( ...
                truth, "longitudinalTireForce", NaN(1,4));
            obj.brakeTorque(row,:) = obj.fieldOrDefault( ...
                truth, "brakeTorque", zeros(1,4));
            obj.roadFriction(row,:) = obj.fieldOrDefault( ...
                truth, "roadFriction", NaN(1,4));
            obj.wheelLocked(row,:) = obj.fieldOrDefault( ...
                truth, "wheelLocked", false(1,4));
            obj.wheelSpeedECU(row,:) = measurements.wheelSpeedMeasured;
            obj.ecuValid(row,:) = measurements.valid;
            obj.ecuStatus(row,:) = measurements.status;

            valid = measurements.valid & isfinite(measurements.wheelSpeedMeasured);
            obj.signedError(row,valid) = ...
                measurements.wheelSpeedMeasured(valid) - truth.wheelSpeedTrue(valid);

            for index = 1:numel(obj.WHEELS)
                wheel = obj.WHEELS(index);
                signal = frame.sensorSignals.(wheel);
                obj.signalLevel(row,index) = signal.level;
                obj.edgesThisStep(row,index) = signal.edgesThisStep;
                obj.droppedEdgesThisStep(row,index) = ...
                    obj.fieldOrDefault(signal, "droppedEdgesThisStep", 0);
                obj.faultActive(row,index) = ...
                    obj.fieldOrDefault(signal, "faultActive", false);
                obj.faultType(row,index) = string( ...
                    obj.fieldOrDefault(signal, "faultType", "none"));
                obj.faultSeverity(row,index) = ...
                    obj.fieldOrDefault(signal, "faultSeverity", 0);
            end
        end

        function result = toTable(obj)
            rows = (1:obj.count).';
            n = obj.count;
            experimentName = repmat(string(obj.config.meta.name), n, 1);
            scenarioType = repmat(string(obj.config.scenario.type), n, 1);
            seed = repmat(obj.config.getSeed(), n, 1);
            errorMode = repmat(string(obj.config.sensorErrors.mode), n, 1);
            latencyMs = repmat(obj.config.sensorErrors.latencyMs, n, 1);
            absoluteJitterUs = repmat(obj.config.sensorErrors.absoluteJitterUs, n, 1);
            relativeJitterPercent = repmat( ...
                obj.config.sensorErrors.relativeJitterPercent, n, 1);

            result = table( ...
                experimentName, scenarioType, seed, ...
                obj.time(rows), obj.throttle(rows), obj.brake(rows), ...
                obj.steering(rows), obj.vehicleSpeed(rows), obj.yawRate(rows), ...
                obj.accelerationX(rows), obj.accelerationY(rows), ...
                errorMode, latencyMs, absoluteJitterUs, relativeJitterPercent, ...
                'VariableNames', { ...
                'experiment_name', 'scenario_type', 'seed', ...
                'time_s', 'throttle', 'brake', 'steering_rad', ...
                'vehicle_speed_mps', 'yaw_rate_radps', ...
                'acceleration_x_mps2', 'acceleration_y_mps2', ...
                'sensor_error_mode', 'latency_ms', ...
                'absolute_jitter_us', 'relative_jitter_percent'});

            for index = 1:numel(obj.WHEELS)
                wheel = obj.WHEELS(index);
                result.("wheel_speed_true_" + wheel + "_mps") = ...
                    obj.wheelSpeedTrue(rows,index);
                result.("wheel_ground_speed_" + wheel + "_mps") = ...
                    obj.wheelGroundSpeed(rows,index);
                result.("wheel_peripheral_speed_" + wheel + "_mps") = ...
                    obj.wheelPeripheralSpeed(rows,index);
                result.("wheel_angular_acceleration_" + wheel + "_radps2") = ...
                    obj.wheelAngularAcceleration(rows,index);
                result.("wheel_slip_ratio_" + wheel) = ...
                    obj.wheelSlipRatio(rows,index);
                result.("normal_load_" + wheel + "_N") = ...
                    obj.normalLoad(rows,index);
                result.("longitudinal_tire_force_" + wheel + "_N") = ...
                    obj.longitudinalTireForce(rows,index);
                result.("brake_torque_" + wheel + "_Nm") = ...
                    obj.brakeTorque(rows,index);
                result.("road_friction_" + wheel) = ...
                    obj.roadFriction(rows,index);
                result.("wheel_locked_" + wheel) = ...
                    obj.wheelLocked(rows,index);
                result.("wheel_speed_ecu_" + wheel + "_mps") = ...
                    obj.wheelSpeedECU(rows,index);
                result.("error_ecu_minus_true_" + wheel + "_mps") = ...
                    obj.signedError(rows,index);
                result.("ecu_valid_" + wheel) = obj.ecuValid(rows,index);
                result.("ecu_status_" + wheel) = obj.ecuStatus(rows,index);
                result.("signal_level_" + wheel) = obj.signalLevel(rows,index);
                result.("edges_this_step_" + wheel) = obj.edgesThisStep(rows,index);
                result.("dropped_edges_" + wheel) = ...
                    obj.droppedEdgesThisStep(rows,index);
                result.("fault_active_" + wheel) = obj.faultActive(rows,index);
                result.("fault_type_" + wheel) = obj.faultType(rows,index);
                result.("fault_severity_" + wheel) = obj.faultSeverity(rows,index);
            end
        end

        function outputPath = exportCsv(obj, outputFolder)
            arguments
                obj
                outputFolder (1,1) string
            end
            if ~isfolder(outputFolder)
                mkdir(outputFolder);
            end

            safeName = regexprep(string(obj.config.meta.name), "[^A-Za-z0-9_-]", "_");
            timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
            fileName = sprintf("%s_seed%d_%s.csv", ...
                safeName, obj.config.getSeed(), timestamp);
            outputPath = fullfile(outputFolder, fileName);
            writetable(obj.toTable(), outputPath);
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
