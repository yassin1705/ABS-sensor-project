classdef DriverCommand
    % DRIVERCOMMAND  Structure normalisee transmise par l'interface conducteur.
    %
    % Champs (cf. doc de cadrage, section 3) :
    %   throttle     - commande d'accelerateur, [0, 1]
    %   brake        - commande de freinage, [0, 1]
    %   steering     - angle volant (rad)
    %   initialSpeed - vitesse initiale du vehicule (m/s), utilisee une seule
    %                  fois au demarrage ; la vitesse reelle est ensuite
    %                  calculee par VehicleDynamics.

    properties
        throttle     (1,1) double {mustBeInRange(throttle, 0, 1)} = 0
        brake        (1,1) double {mustBeInRange(brake, 0, 1)} = 0
        steering     (1,1) double = 0
        initialSpeed (1,1) double {mustBeNonnegative} = 0
    end

    methods
        function obj = DriverCommand(throttle, brake, steering, initialSpeed)
            arguments
                throttle     (1,1) double = 0
                brake        (1,1) double = 0
                steering     (1,1) double = 0
                initialSpeed (1,1) double = 0
            end
            obj.throttle     = throttle;
            obj.brake        = brake;
            obj.steering     = steering;
            obj.initialSpeed = initialSpeed;

            if throttle > 0 && brake > 0
                warning("DriverCommand:conflictingInputs", ...
                    "throttle et brake actifs simultanement (%.2f / %.2f).", ...
                    throttle, brake);
            end
        end

        function s = toStruct(obj)
            % Conversion en struct simple, utile pour les bus Simulink
            % ou la journalisation.
            s = struct( ...
                "throttle",     obj.throttle, ...
                "brake",        obj.brake, ...
                "steering",     obj.steering, ...
                "initialSpeed", obj.initialSpeed);
        end
    end

    methods (Static)
        function obj = fromStruct(s)
            obj = DriverCommand(s.throttle, s.brake, s.steering, s.initialSpeed);
        end
    end
end