classdef ScenarioGenerator < handle
    % SCENARIOGENERATOR  Transforme une configuration de scenario en
    % DriverCommand deterministe pour un instant donne.
    %
    % Deux formes sont supportees :
    %   - les scenarios nommes straight_braking et constant_command ;
    %   - une liste generique de segments temporels, extensible sans
    %     modifier le controleur de simulation.

    properties (SetAccess = private)
        scenario struct
        initialSpeed (1,1) double
    end

    methods
        function obj = ScenarioGenerator(scenario)
            arguments
                scenario (1,1) struct
            end
            assert(isfield(scenario, "initialSpeed_kmh"), ...
                "Le scenario doit contenir initialSpeed_kmh.");
            obj.scenario = scenario;
            obj.initialSpeed = scenario.initialSpeed_kmh / 3.6;
        end

        function command = commandAt(obj, time)
            arguments
                obj
                time (1,1) double {mustBeNonnegative}
            end

            if isfield(obj.scenario, "segments")
                values = obj.commandFromSegments(time);
            else
                values = obj.commandFromNamedScenario(time);
            end

            command = DriverCommand(values.throttle, values.brake, ...
                values.steering, obj.initialSpeed);
        end
    end

    methods (Access = private)
        function values = commandFromNamedScenario(obj, time)
            type = string(obj.scenario.type);
            switch type
                case "straight_braking"
                    if time < obj.scenario.brakeStartTime
                        throttle = obj.scenario.throttleCommand;
                        brake = 0;
                    else
                        throttle = 0;
                        brake = obj.scenario.brakeCommand;
                    end
                    steering = obj.scenario.steeringCommand;

                case "constant_command"
                    throttle = obj.scenario.throttleCommand;
                    brake = obj.scenario.brakeCommand;
                    steering = obj.scenario.steeringCommand;

                otherwise
                    error("ScenarioGenerator:unsupportedType", ...
                        "Type de scenario non supporte : %s.", type);
            end

            values = struct("throttle", throttle, "brake", brake, ...
                "steering", steering);
        end

        function values = commandFromSegments(obj, time)
            segments = obj.scenario.segments;
            values = struct("throttle", 0, "brake", 0, "steering", 0);
            matched = false;

            for index = 1:numel(segments)
                segment = segments(index);
                assert(isfield(segment, "startTime") && ...
                    isfield(segment, "endTime"), ...
                    "Chaque segment doit contenir startTime et endTime.");
                if time >= segment.startTime && time < segment.endTime
                    values.throttle = obj.fieldOrDefault(segment, "throttle", 0);
                    values.brake = obj.fieldOrDefault(segment, "brake", 0);
                    values.steering = obj.fieldOrDefault(segment, "steering", 0);
                    matched = true;
                    break;
                end
            end

            if ~matched && isfield(obj.scenario, "defaultCommand")
                default = obj.scenario.defaultCommand;
                values.throttle = obj.fieldOrDefault(default, "throttle", 0);
                values.brake = obj.fieldOrDefault(default, "brake", 0);
                values.steering = obj.fieldOrDefault(default, "steering", 0);
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
