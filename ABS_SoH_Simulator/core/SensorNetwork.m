classdef SensorNetwork < handle
    % SENSORNETWORK  Reseau des quatre capteurs actifs ABS sains.
    %
    % Entree : VehicleState issu de VehicleDynamics.
    % Sortie : un signal brut evenementiel par roue. Aucun ECU, filtre,
    % calcul de vitesse ni defaut n'est implemente dans cette classe.

    properties (SetAccess = private)
        % Les instances sont construites dans le constructeur a partir de
        % la configuration. Aucun capteur par defaut n'est physiquement
        % significatif, donc ces proprietes ne sont pas typees ici.
        FL
        FR
        RL
        RR
    end

    methods
        function obj = SensorNetwork(sensorConfig)
            arguments
                sensorConfig (1,1) struct
            end

            requiredWheels = ["FL", "FR", "RL", "RR"];
            assert(isfield(sensorConfig, "edgesPerRevolution"), ...
                "La configuration capteur doit contenir edgesPerRevolution.");

            for wheel = requiredWheels
                assert(isfield(sensorConfig.edgesPerRevolution, wheel), ...
                    "Configuration manquante pour la roue %s.", wheel);
            end

            edges = sensorConfig.edgesPerRevolution;
            obj.FL = WheelSensor("FL", edges.FL);
            obj.FR = WheelSensor("FR", edges.FR);
            obj.RL = WheelSensor("RL", edges.RL);
            obj.RR = WheelSensor("RR", edges.RR);
        end

        function reset(obj)
            obj.FL.reset();
            obj.FR.reset();
            obj.RL.reset();
            obj.RR.reset();
        end

        function signals = step(obj, vehicleState)
            arguments
                obj
                vehicleState (1,1) struct
            end
            assert(isfield(vehicleState, "wheelAngleTotal"), ...
                "VehicleState doit contenir wheelAngleTotal.");

            theta = vehicleState.wheelAngleTotal;
            signals = struct( ...
                "time", vehicleState.time, ...
                "FL", obj.FL.step(vehicleState.time, theta(1)), ...
                "FR", obj.FR.step(vehicleState.time, theta(2)), ...
                "RL", obj.RL.step(vehicleState.time, theta(3)), ...
                "RR", obj.RR.step(vehicleState.time, theta(4)));
        end
    end
end
