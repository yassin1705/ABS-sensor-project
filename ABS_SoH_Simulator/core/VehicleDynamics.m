classdef VehicleDynamics < handle
    % VEHICLEDYNAMICS  Dynamique vehicule avec deux niveaux de fidelite.
    %
    % model = "legacy"
    %   Conserve exactement le modele bicyclette historique sans glissement.
    %
    % model = "longitudinal_wheel_slip"
    %   Ajoute quatre vitesses angulaires independantes, les charges
    %   verticales, les couples de freinage, les forces pneu-route, le
    %   glissement longitudinal et le blocage. L'ABS reste desactive.

    properties (SetAccess = private)
        mass double
        wheelRadius double
        track double
        wheelbase double
        muRoad double
        yawInertia double

        dynamicsModel (1,1) string = "legacy"
        dynamics struct
        brakeSystem struct

        speed double = 0
        yawRate double = 0
        time double = 0
        wheelAngleTotal (1,4) double = zeros(1,4)
        wheelOmegaState (1,4) double = zeros(1,4)
        lastAccelerationX double = 0
        initialized logical = false
    end

    properties (Constant, Access = private)
        LEGACY_MAX_ACCEL = 4.0
        LEGACY_MAX_DECEL = 9.0
        G = 9.81
        WHEELS = ["FL", "FR", "RL", "RR"]
    end

    methods
        function obj = VehicleDynamics(vehicleParams, dynamicsParams, ...
                brakeParams, roadParams)
            arguments
                vehicleParams (1,1) struct
                dynamicsParams (1,1) struct = struct("model", "legacy")
                brakeParams (1,1) struct = struct()
                roadParams (1,1) struct = struct()
            end

            obj.mass = vehicleParams.mass;
            obj.wheelRadius = vehicleParams.wheelRadius;
            obj.track = vehicleParams.track;
            obj.wheelbase = vehicleParams.wheelbase;
            obj.muRoad = vehicleParams.muRoad;
            obj.yawInertia = vehicleParams.yawInertia;

            if isfield(dynamicsParams, "model")
                obj.dynamicsModel = string(dynamicsParams.model);
            end
            obj.dynamics = dynamicsParams;
            obj.brakeSystem = brakeParams;
            if isfield(roadParams, "muRoad")
                obj.muRoad = roadParams.muRoad;
            end
        end

        function initialize(obj, driverCommand)
            arguments
                obj
                driverCommand (1,1) struct
            end
            obj.speed = driverCommand.initialSpeed;
            obj.yawRate = 0;
            obj.time = 0;
            obj.wheelAngleTotal = zeros(1,4);
            obj.wheelOmegaState = repmat( ...
                driverCommand.initialSpeed / obj.wheelRadius, 1, 4);
            obj.lastAccelerationX = 0;
            obj.initialized = true;
        end

        function state = step(obj, driverCommand, dt)
            arguments
                obj
                driverCommand (1,1) struct
                dt (1,1) double {mustBePositive}
            end

            if ~obj.initialized
                obj.initialize(driverCommand);
            end

            if driverCommand.brake > 0 && driverCommand.throttle > 0
                driverCommand.throttle = 0;
            end

            switch obj.dynamicsModel
                case "legacy"
                    state = obj.stepLegacy(driverCommand, dt);
                case "longitudinal_wheel_slip"
                    state = obj.stepWheelSlip(driverCommand, dt);
                otherwise
                    error("VehicleDynamics:unsupportedModel", ...
                        "Modele dynamique non supporte : %s.", ...
                        obj.dynamicsModel);
            end
        end
    end

    methods (Access = private)
        function state = stepLegacy(obj, driverCommand, dt)
            targetSpeed = driverCommand.throttle * (100 / 3.6);

            if driverCommand.brake > 0
                aBrakeMax = min(obj.LEGACY_MAX_DECEL, obj.muRoad * obj.G);
                accelX = -driverCommand.brake * aBrakeMax;
            else
                speedGap = targetSpeed - obj.speed;
                if speedGap >= 0
                    accelX = min(obj.LEGACY_MAX_ACCEL, speedGap / dt);
                else
                    accelX = max(-1.0, speedGap / dt);
                end
            end

            obj.speed = max(0, obj.speed + accelX * dt);
            obj.yawRate = obj.computeYawRate(driverCommand.steering);
            accelY = obj.speed * obj.yawRate;
            wheelGroundSpeed = obj.localWheelGroundSpeeds();
            wheelOmega = wheelGroundSpeed / obj.wheelRadius;

            obj.wheelOmegaState = wheelOmega;
            obj.wheelAngleTotal = obj.wheelAngleTotal + wheelOmega * dt;
            obj.time = obj.time + dt;
            obj.lastAccelerationX = accelX;

            staticLoads = obj.staticNormalLoads();
            zeros4 = zeros(1,4);
            state = obj.buildState(driverCommand, accelX, accelY, ...
                wheelGroundSpeed, wheelGroundSpeed, wheelOmega, zeros4, ...
                zeros4, staticLoads, zeros4, zeros4, false(1,4));
        end

        function state = stepWheelSlip(obj, driverCommand, dt)
            % Sous-pas courts pour stabiliser le couplage raide pneu/roue.
            integrationStep = 0.001;
            substepCount = max(1, ceil(dt / integrationStep));
            h = dt / substepCount;

            finalTireForce = zeros(1,4);
            finalNormalLoad = obj.staticNormalLoads();
            finalBrakeTorque = zeros(1,4);
            finalWheelAlpha = zeros(1,4);
            finalAccelX = obj.lastAccelerationX;

            for substepIndex = 1:substepCount
                obj.yawRate = obj.computeYawRate(driverCommand.steering);
                groundSpeed = obj.localWheelGroundSpeeds();
                normalLoad = obj.dynamicNormalLoads(obj.lastAccelerationX);
                peripheralSpeed = obj.wheelRadius * obj.wheelOmegaState;

                denominator = max(abs(groundSpeed), ...
                    obj.dynamics.lowSpeedRegularization);
                slip = (peripheralSpeed - groundSpeed) ./ denominator;

                frictionLimit = obj.muRoad * normalLoad;
                tireForce = frictionLimit .* tanh( ...
                    obj.dynamics.slipShapeFactor * slip);

                driveTorque = obj.driveTorque(driverCommand.throttle);
                brakeTorque = obj.brakeTorque(driverCommand.brake);

                wheelAlpha = (driveTorque - brakeTorque - ...
                    obj.wheelRadius * tireForce) / ...
                    obj.dynamics.wheelInertia;

                nextOmega = obj.wheelOmegaState + wheelAlpha * h;
                % La V1 ne modelise pas la marche arriere. Un frein ne
                % peut donc pas inverser artificiellement une roue.
                obj.wheelOmegaState = max(0, nextOmega);
                obj.wheelAngleTotal = obj.wheelAngleTotal + ...
                    obj.wheelOmegaState * h;

                resistance = obj.longitudinalResistance();
                accelX = (sum(tireForce) - resistance) / obj.mass;
                obj.speed = max(0, obj.speed + accelX * h);
                obj.lastAccelerationX = accelX;

                finalTireForce = tireForce;
                finalNormalLoad = normalLoad;
                finalBrakeTorque = brakeTorque;
                finalWheelAlpha = wheelAlpha;
                finalAccelX = accelX;
            end

            obj.yawRate = obj.computeYawRate(driverCommand.steering);
            finalGroundSpeed = obj.localWheelGroundSpeeds();
            finalPeripheralSpeed = obj.wheelRadius * obj.wheelOmegaState;
            denominator = max(abs(finalGroundSpeed), ...
                obj.dynamics.lowSpeedRegularization);
            finalSlip = (finalPeripheralSpeed - finalGroundSpeed) ./ denominator;

            wheelLocked = driverCommand.brake > 0 & ...
                finalGroundSpeed > obj.dynamics.minimumLockSpeed & ...
                finalSlip <= -obj.dynamics.lockSlipThreshold;

            obj.time = obj.time + dt;
            accelY = obj.speed * obj.yawRate;
            state = obj.buildState(driverCommand, finalAccelX, accelY, ...
                finalGroundSpeed, finalPeripheralSpeed, ...
                obj.wheelOmegaState, finalWheelAlpha, finalSlip, ...
                finalNormalLoad, finalTireForce, finalBrakeTorque, wheelLocked);
        end

        function yawRate = computeYawRate(obj, steering)
            if obj.speed > 0.1
                yawRate = obj.speed * tan(steering) / obj.wheelbase;
            else
                yawRate = 0;
            end
        end

        function speeds = localWheelGroundSpeeds(obj)
            halfDelta = 0.5 * obj.yawRate * obj.track;
            speeds = [obj.speed - halfDelta, obj.speed + halfDelta, ...
                obj.speed - halfDelta, obj.speed + halfDelta];
            speeds = max(speeds, 0);
        end

        function loads = staticNormalLoads(obj)
            if isfield(obj.dynamics, "distanceCGToFrontAxle")
                lf = obj.dynamics.distanceCGToFrontAxle;
                lr = obj.wheelbase - lf;
            else
                lf = obj.wheelbase / 2;
                lr = obj.wheelbase / 2;
            end
            frontAxle = obj.mass * obj.G * lr / obj.wheelbase;
            rearAxle = obj.mass * obj.G * lf / obj.wheelbase;
            loads = [frontAxle/2, frontAxle/2, rearAxle/2, rearAxle/2];
        end

        function loads = dynamicNormalLoads(obj, accelerationX)
            staticLoads = obj.staticNormalLoads();
            transfer = -obj.mass * accelerationX * ...
                obj.dynamics.cgHeight / obj.wheelbase;
            frontAxle = 2 * staticLoads(1) + transfer;
            rearAxle = 2 * staticLoads(3) - transfer;

            minimumAxleLoad = 0.05 * obj.mass * obj.G;
            frontAxle = max(minimumAxleLoad, frontAxle);
            rearAxle = max(minimumAxleLoad, rearAxle);
            scale = obj.mass * obj.G / (frontAxle + rearAxle);
            frontAxle = frontAxle * scale;
            rearAxle = rearAxle * scale;
            loads = [frontAxle/2, frontAxle/2, rearAxle/2, rearAxle/2];
        end

        function torque = driveTorque(obj, throttle)
            total = throttle * obj.dynamics.maxDriveTorque;
            torque = repmat(total / 4, 1, 4);
        end

        function torque = brakeTorque(obj, brake)
            total = brake * obj.brakeSystem.maxBrakeTorque;
            front = total * obj.brakeSystem.frontDistribution / 2;
            rear = total * (1 - obj.brakeSystem.frontDistribution) / 2;
            torque = [front, front, rear, rear];
        end

        function resistance = longitudinalResistance(obj)
            if obj.speed <= eps
                resistance = 0;
                return;
            end
            rolling = obj.dynamics.rollingResistanceCoefficient * ...
                obj.mass * obj.G;
            aerodynamic = obj.dynamics.aerodynamicDragCoefficient * ...
                obj.speed^2;
            resistance = rolling + aerodynamic;
        end

        function state = buildState(obj, driverCommand, accelX, accelY, ...
                wheelGroundSpeed, wheelPeripheralSpeed, wheelOmega, ...
                wheelAngularAcceleration, wheelSlipRatio, normalLoad, ...
                tireForce, brakeTorque, wheelLocked)
            state = struct( ...
                "time", obj.time, ...
                "vehicleSpeed", obj.speed, ...
                "yawRate", obj.yawRate, ...
                "accelerationX", accelX, ...
                "accelerationY", accelY, ...
                "wheelGroundSpeed", wheelGroundSpeed, ...
                "wheelPeripheralSpeed", wheelPeripheralSpeed, ...
                "wheelSpeedTrue", wheelPeripheralSpeed, ...
                "wheelOmega", wheelOmega, ...
                "wheelAngularAcceleration", wheelAngularAcceleration, ...
                "wheelAngle", mod(obj.wheelAngleTotal, 2*pi), ...
                "wheelAngleTotal", obj.wheelAngleTotal, ...
                "wheelSlipRatio", wheelSlipRatio, ...
                "normalLoad", normalLoad, ...
                "longitudinalTireForce", tireForce, ...
                "brakeTorque", brakeTorque, ...
                "roadFriction", repmat(obj.muRoad, 1, 4), ...
                "wheelLocked", wheelLocked, ...
                "brakeState", struct( ...
                    "braking", driverCommand.brake > 0, ...
                    "brakeCommand", driverCommand.brake, ...
                    "absActive", false(1,4)));
        end
    end
end
