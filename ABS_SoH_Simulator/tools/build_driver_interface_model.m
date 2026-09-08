function build_driver_interface_model()
% BUILD_DRIVER_INTERFACE_MODEL
%   Construit programmatiquement le modele Simulink "DriverInterface.slx"
%   avec 4 sliders (throttle, brake, steering, initialSpeed) exposes via
%   un bus DriverCommand en sortie.
%
%   A executer UNE FOIS pour generer le .slx, qui sera ensuite versionne
%   normalement (fichier binaire, donc pas de diff Git lisible mais c'est
%   attendu pour du Simulink).
%
%   Usage :
%       build_driver_interface_model();
%       open_system("DriverInterface");

    modelName = "DriverInterface";

    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    new_system(modelName);
    open_system(modelName);

    % --- Bus object DriverCommand -----------------------------------
    elems(1) = Simulink.BusElement;
    elems(1).Name = "throttle";
    elems(1).DataType = "double";

    elems(2) = Simulink.BusElement;
    elems(2).Name = "brake";
    elems(2).DataType = "double";

    elems(3) = Simulink.BusElement;
    elems(3).Name = "steering";
    elems(3).DataType = "double";

    elems(4) = Simulink.BusElement;
    elems(4).Name = "initialSpeed";
    elems(4).DataType = "double";

    driverCommandBus = Simulink.Bus;
    driverCommandBus.Elements = elems;
    assignin("base", "DriverCommandBus", driverCommandBus);

    % --- 4 Slider Gain blocks (via Constant + Slider Switch equivalent) ---
    % Simulink ne fournit pas nativement un "slider" en bloc de simulation ;
    % on utilise un bloc "Slider Gain" applique a une constante 1, ce qui
    % est le pattern standard pour piloter une valeur scalaire a la main
    % pendant la simulation (double-cliquer dessus ouvre le curseur).

    names  = ["Throttle", "Brake", "Steering", "InitialSpeed"];
    ranges = [0 1; 0 1; -0.6 0.6; 0 40];   % steering en rad, vitesse en m/s
    yPos   = [50, 150, 250, 350];

    for k = 1:numel(names)
        constBlock = modelName + "/Const_" + names(k);
        sliderBlock = modelName + "/Slider_" + names(k);

        add_block("simulink/Sources/Constant", constBlock, ...
            "Value", "1", ...
            "Position", [50, yPos(k), 100, yPos(k)+30]);

        add_block("simulink/Math Operations/Slider Gain", sliderBlock, ...
            "Position", [150, yPos(k), 260, yPos(k)+30], ...
            "Low", num2str(ranges(k,1)), ...
            "High", num2str(ranges(k,2)), ...
            "Gain", num2str(mean(ranges(k,:))));

        add_line(modelName, "Const_" + names(k) + "/1", "Slider_" + names(k) + "/1");
    end

    % --- Bus Creator pour assembler DriverCommand -------------------
    busCreator = modelName + "/DriverCommand_BusCreator";
    add_block("simulink/Signal Routing/Bus Creator", busCreator, ...
        "Inputs", "4", ...
        "Position", [320, 40, 370, 360]);

    for k = 1:numel(names)
        add_line(modelName, "Slider_" + names(k) + "/1", ...
            "DriverCommand_BusCreator/" + k);
    end

    % --- Out1 (sortie du sous-systeme, vers SimulationController) ---
    add_block("simulink/Sinks/Out1", modelName + "/DriverCommand_Out", ...
        "Position", [420, 180, 450, 210]);
    add_line(modelName, "DriverCommand_BusCreator/1", "DriverCommand_Out/1");

    % --- Sauvegarde ---------------------------------------------------
    % Le modele runtime vit dans core/, a cote de VehicleDynamics.m,
    % meme si ce script de generation vit dans tools/.
    projectRoot = fileparts(fileparts(mfilename("fullpath")));  % ABS_SoH_Project/
    outputPath  = fullfile(projectRoot, "core", modelName + ".slx");

    save_system(modelName, outputPath);
    fprintf("Modele '%s.slx' cree avec succes dans %s\n", modelName, outputPath);
    fprintf("Double-cliquez sur chaque bloc 'Slider Gain' pour piloter la commande pendant la simulation.\n");
end