classdef SimulationController < handle
    % SIMULATIONCONTROLLER  Orchestre un pas complet de simulation.
    %
    % La chaine physique reste stable : scenario -> dynamique -> capteurs
    % -> injection -> ECU. Des modules d'analyse peuvent etre ajoutes via
    % addAnalysisModule(), ce qui permettra d'integrer WheelSpeedEstimator
    % sans modifier les classes physiques.

    properties (SetAccess = private)
        config
        scenarioGenerator
        vehicleDynamics
        sensorNetwork
        errorInjector
        ecu
    end

    properties (Access = private)
        analysisModules cell = {}
    end

    methods
        function obj = SimulationController(config)
            arguments
                config (1,1) SimulationConfig
            end
            obj.config = config;
            obj.scenarioGenerator = ScenarioGenerator(config.scenario);
            obj.vehicleDynamics = VehicleDynamics(config.vehicle, ...
                config.dynamics, config.brakeSystem, config.road);
            obj.sensorNetwork = SensorNetwork(config.sensor);
            obj.errorInjector = SensorErrorInjector(config.sensorErrors, config.faults);
            obj.ecu = ABSECU(config.ecu, config.sensor, config.vehicle.wheelRadius);
            obj.reset();
        end

        function reset(obj)
            initialCommand = obj.scenarioGenerator.commandAt(0);
            obj.vehicleDynamics.initialize(initialCommand.toStruct());
            obj.sensorNetwork.reset();
            obj.errorInjector.reset();
            obj.ecu.reset();
        end

        function addAnalysisModule(obj, name, module)
            arguments
                obj
                name (1,1) string
                module
            end
            assert(isvarname(char(name)), ...
                "Le nom du module d'analyse doit etre un identifiant MATLAB valide.");
            obj.analysisModules{end + 1} = struct("name", name, "module", module);
        end

        function frame = stepAt(obj, commandTime)
            arguments
                obj
                commandTime (1,1) double {mustBeNonnegative}
            end

            command = obj.scenarioGenerator.commandAt(commandTime);
            truth = obj.vehicleDynamics.step(command.toStruct(), ...
                obj.config.simulation.dt);
            cleanSignals = obj.sensorNetwork.step(truth);
            sensorSignals = obj.errorInjector.step(cleanSignals, truth);
            measurements = obj.ecu.step(sensorSignals);

            frame = struct( ...
                "commandTime", commandTime, ...
                "command", command.toStruct(), ...
                "truth", truth, ...
                "cleanSensorSignals", cleanSignals, ...
                "sensorSignals", sensorSignals, ...
                "ecuMeasurements", measurements, ...
                "analysis", struct());

            for index = 1:numel(obj.analysisModules)
                entry = obj.analysisModules{index};
                frame.analysis.(entry.name) = entry.module.step(frame);
            end
        end
    end
end
