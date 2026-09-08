classdef WheelSensor < handle
    % WHEELSENSOR  Modele sain d'un capteur actif ABS.
    %
    % Le capteur ne calcule pas une vitesse. Il transforme la rotation
    % mecanique de la roue en un signal numerique : une suite de fronts
    % horodates associes au passage des poles de la bague magnetique.
    %
    % La vitesse ECU sera reconstruite plus tard a partir de ces fronts.

    properties (SetAccess = private)
        wheelName (1,1) string
        edgesPerRevolution (1,1) double
        anglePerEdge (1,1) double
        previousTime (1,1) double = 0
        previousAngleTotal (1,1) double = 0
        totalEdges (1,1) double = 0
        currentLevel (1,1) logical = false
    end

    methods
        function obj = WheelSensor(wheelName, edgesPerRevolution)
            arguments
                wheelName (1,1) string
                edgesPerRevolution (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.wheelName = wheelName;
            obj.edgesPerRevolution = edgesPerRevolution;
            obj.anglePerEdge = 2*pi / edgesPerRevolution;
        end

        function reset(obj)
            obj.previousTime = 0;
            obj.previousAngleTotal = 0;
            obj.totalEdges = 0;
            obj.currentLevel = false;
        end

        function signal = step(obj, time, angleTotal)
            % STEP Produit tous les fronts apparus entre le dernier appel
            % et l'instant courant. Les horodatages sont interpoles dans
            % le pas de simulation, ce qui evite de perdre des fronts a
            % grande vitesse.
            arguments
                obj
                time (1,1) double {mustBeNonnegative}
                angleTotal (1,1) double
            end

            if time < obj.previousTime
                error("WheelSensor:timeRegression", ...
                    "Le temps doit etre croissant pour le capteur %s.", obj.wheelName);
            end

            theta0 = obj.previousAngleTotal;
            theta1 = angleTotal;
            edgeIndex0 = floor(theta0 / obj.anglePerEdge);
            edgeIndex1 = floor(theta1 / obj.anglePerEdge);

            if edgeIndex1 > edgeIndex0
                crossedIndices = (edgeIndex0 + 1):edgeIndex1;
            elseif edgeIndex1 < edgeIndex0
                % Support du sens inverse, meme si VehicleDynamics V1 ne
                % simule actuellement que le roulement vers l'avant.
                crossedIndices = edgeIndex0:-1:(edgeIndex1 + 1);
            else
                crossedIndices = zeros(1,0);
            end

            nEdges = numel(crossedIndices);
            edgeTimes = zeros(1, nEdges);
            edgeLevels = false(1, nEdges);

            if nEdges > 0
                deltaTheta = theta1 - theta0;
                deltaTime = time - obj.previousTime;
                if deltaTheta == 0
                    error("WheelSensor:inconsistentAngle", ...
                        "Franchissement de front impossible sans variation d'angle.");
                end

                for k = 1:nEdges
                    thetaEdge = crossedIndices(k) * obj.anglePerEdge;
                    ratio = (thetaEdge - theta0) / deltaTheta;
                    edgeTimes(k) = obj.previousTime + ratio * deltaTime;

                    if edgeIndex1 > edgeIndex0
                        % Niveau logique apres un front dans le sens avant.
                        edgeLevels(k) = logical(mod(crossedIndices(k), 2));
                    else
                        % Niveau logique apres un front dans le sens inverse.
                        edgeLevels(k) = logical(mod(crossedIndices(k) - 1, 2));
                    end
                end
            end

            obj.totalEdges = obj.totalEdges + nEdges;
            obj.currentLevel = logical(mod(edgeIndex1, 2));
            obj.previousTime = time;
            obj.previousAngleTotal = angleTotal;

            signal = struct( ...
                "wheel", obj.wheelName, ...
                "time", time, ...
                "level", obj.currentLevel, ...
                "edgeTimes", edgeTimes, ...
                "edgeLevels", edgeLevels, ...
                "edgesThisStep", nEdges, ...
                "totalEdges", obj.totalEdges, ...
                "edgesPerRevolution", obj.edgesPerRevolution, ...
                "healthy", true);
        end
    end
end
