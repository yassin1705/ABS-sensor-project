function SimulationApp()
% SIMULATIONAPP  Fenetre interactive pilotee au clavier, simulant des
% pedales reelles (accelerateur/frein) et un volant auto-centre, avec
% affichage live de la vitesse des 4 roues et du signal ABS sain.
%
% Usage :
%   SimulationApp()
%
% Commandes clavier (maintenir la touche enfoncee) :
%   Fleche HAUT   -> accelerateur : monte pendant l'appui, redescend au relachement
%   Fleche BAS    -> frein        : monte pendant l'appui, redescend au relachement
%   Fleche GAUCHE -> volant vers la gauche, revient a 0 au relachement
%   Fleche DROITE -> volant vers la droite, revient a 0 au relachement
%   R             -> reset de la simulation
%
% Notes :
%   - L'accelerateur (0-100%) ne fixe pas la vitesse directement : il
%     definit une vitesse cible que le vehicule rejoint sous acceleration
%     bornee (cf. VehicleDynamics.step). 100% -> cible 100 km/h.
%   - Le frein est prioritaire sur l'accelerateur (coherent avec
%     VehicleDynamics : brake>0 desactive throttle).
%   - Les sliders affiches sont des jauges en lecture seule ; la vraie
%     commande vient du clavier, pour simuler l'effet "pedale relachee
%     = retour progressif" et "volant relache = auto-centrage".
%   - L'app affiche le signal brut evenementiel produit par SensorNetwork.
%     Elle ne contient pas encore d'ECU ni de reconstruction de vitesse.

    projectRoot = fileparts(fileparts(mfilename("fullpath")));  % ABS_SoH_Project/
    addpath(fullfile(projectRoot, "core"));

    warning("off", "DriverCommand:conflictingInputs");

    % Nettoyage defensif : si une session precedente a plante avant
    % d'atteindre CloseRequestFcn, un timer orphelin peut rester actif
    % et generer des erreurs "Unable to find function" au tick suivant.
    oldTimers = timerfindall("Name", "SimulationApp_loop");
    if ~isempty(oldTimers)
        stop(oldTimers);
        delete(oldTimers);
    end

    % --- Parametres charges depuis le scenario de reference ------------
    % Les commandes restent interactives, mais les parametres physiques,
    % capteurs, incertitudes et ECU sont centralises dans le JSON.
    configPath = fullfile(projectRoot, "scenarios", "scenario_reference.json");
    cfg = SimulationConfig(configPath);
    vehicleParams = cfg.vehicle;
    vd = VehicleDynamics(vehicleParams);
    vd.initialize(DriverCommand(0, 0, 0, 0).toStruct());

    sensorConfig = cfg.sensor;
    sensors = SensorNetwork(sensorConfig);
    sensorErrorConfig = cfg.sensorErrors;
    errorInjector = SensorErrorInjector(sensorErrorConfig, cfg.faults);
    ecuConfig = cfg.ecu;
    ecu = ABSECU(ecuConfig, sensorConfig, vehicleParams.wheelRadius);

    dt = 0.1; % s, pas de la boucle temps reel
    SIGNAL_WINDOW = 0.5; % s, largeur de la fenetre affichee
    wheelNames = ["FL", "FR", "RL", "RR"];
    selectedSignalWheel = "FL";
    signalHistory = emptySignalHistory();
    minErrorKmh = Inf;
    maxErrorKmh = -Inf;

    % --- Etat des pedales/volant (jauges internes, 0-100% / -30..30 deg) ---
    pedalState = struct( ...
        "throttle", 0, "brake", 0, "steering", 0);

    % --- Touches actuellement maintenues ---
    keysHeld = struct("up", false, "down", false, "left", false, "right", false);

    % --- Vitesses de rampe (unites/seconde) ---
    RAMP_THROTTLE_UP   = 80;   % %/s pendant l'appui
    RAMP_THROTTLE_DOWN = 120;  % %/s au relachement (pedale revient plus vite que le pied pousse)
    RAMP_BRAKE_UP      = 150;  % %/s pendant l'appui (freinage plus reactif)
    RAMP_BRAKE_DOWN    = 150;  % %/s au relachement
    RAMP_STEER         = 60;   % deg/s pendant l'appui
    RAMP_STEER_CENTER  = 90;   % deg/s au relachement (auto-centrage, plus rapide)
    STEER_MAX          = 30;   % deg

    % ============================== UI ==================================
    fig = uifigure("Name", "ABS SoH - Interface conducteur (clavier)", ...
        "Position", [100 100 720 720]);

    mainGrid = uigridlayout(fig, [9, 2]);
    mainGrid.RowHeight   = {22, 40, 40, 40, 40, "1x", 28, 28, "2x"};
    mainGrid.ColumnWidth = {150, "1x"};

    lblHelp = uilabel(mainGrid, "Text", ...
        "Fleches HAUT/BAS = accelerateur/frein  |  GAUCHE/DROITE = volant  |  R = reset", ...
        "FontAngle", "italic");
    lblHelp.Layout.Row = 1; lblHelp.Layout.Column = [1 2];

    lblThrottle = uilabel(mainGrid, "Text", "Accelerateur : 0 %");
    lblThrottle.Layout.Row = 2; lblThrottle.Layout.Column = 1;
    sldThrottle = uislider(mainGrid, "Limits", [0 100], "Value", 0, "Enable", "off");
    sldThrottle.Layout.Row = 2; sldThrottle.Layout.Column = 2;

    lblBrake = uilabel(mainGrid, "Text", "Frein : 0 %");
    lblBrake.Layout.Row = 3; lblBrake.Layout.Column = 1;
    sldBrake = uislider(mainGrid, "Limits", [0 100], "Value", 0, "Enable", "off");
    sldBrake.Layout.Row = 3; sldBrake.Layout.Column = 2;

    lblSteer = uilabel(mainGrid, "Text", "Angle volant : 0 deg");
    lblSteer.Layout.Row = 4; lblSteer.Layout.Column = 1;
    sldSteer = uislider(mainGrid, "Limits", [-STEER_MAX STEER_MAX], "Value", 0, "Enable", "off");
    sldSteer.Layout.Row = 4; sldSteer.Layout.Column = 2;

    btnReset = uibutton(mainGrid, "Text", "Reset (R)");
    btnReset.Layout.Row = 5; btnReset.Layout.Column = 1;

    lblVehicleSpeed = uilabel(mainGrid, "Text", "Vitesse vehicule : 0.0 km/h", ...
        "FontWeight", "bold", "FontSize", 14);
    lblVehicleSpeed.Layout.Row = 5; lblVehicleSpeed.Layout.Column = 2;

    wheelPanel = uipanel(mainGrid, "Title", "Vitesse des roues et niveau du signal ABS");
    wheelPanel.Layout.Row = 6; wheelPanel.Layout.Column = [1 2];
    wheelGrid = uigridlayout(wheelPanel, [2, 2]);

    lblFL = uilabel(wheelGrid, "Text", "FL : 0.0 km/h | S=0", "FontSize", 14, "HorizontalAlignment", "center");
    lblFR = uilabel(wheelGrid, "Text", "FR : 0.0 km/h | S=0", "FontSize", 14, "HorizontalAlignment", "center");
    lblRL = uilabel(wheelGrid, "Text", "RL : 0.0 km/h | S=0", "FontSize", 14, "HorizontalAlignment", "center");
    lblRR = uilabel(wheelGrid, "Text", "RR : 0.0 km/h | S=0", "FontSize", 14, "HorizontalAlignment", "center");

    lblErrorRange = uilabel(mainGrid, ...
        "Text", "Ecart ECU - reel : aucune mesure valide", ...
        "FontWeight", "bold");
    lblErrorRange.Layout.Row = 7; lblErrorRange.Layout.Column = [1 2];

    lblSignal = uilabel(mainGrid, "Text", "Signal brut : FL (0 fronts)", "FontWeight", "bold");
    lblSignal.Layout.Row = 8; lblSignal.Layout.Column = 1;
    signalSelector = uidropdown(mainGrid, "Items", cellstr(wheelNames), "Value", "FL");
    signalSelector.Layout.Row = 8; signalSelector.Layout.Column = 2;
    signalSelector.ValueChangedFcn = @(src, ~) selectSignalWheel(string(src.Value));

    signalAxes = uiaxes(mainGrid);
    signalAxes.Layout.Row = 9; signalAxes.Layout.Column = [1 2];
    signalAxes.YLim = [-0.2 1.2];
    signalAxes.YTick = [0 1];
    signalAxes.XLabel.String = "Temps (s)";
    signalAxes.YLabel.String = "Niveau logique";
    signalAxes.Title.String = "Signal ABS brut - apres incertitudes nominales";
    grid(signalAxes, "on");

    % ============================ Clavier ==============================
    fig.KeyPressFcn   = @(~, evt) setKeyState(evt.Key, true);
    fig.KeyReleaseFcn = @(~, evt) setKeyState(evt.Key, false);

    btnReset.ButtonPushedFcn = @(~, ~) resetSimulation();

    % ============================ Boucle temps reel =======================
    t = timer("ExecutionMode", "fixedRate", "Period", dt, ...
        "Name", "SimulationApp_loop", ...
        "TimerFcn", @(~, ~) updateStep());
    start(t);

    fig.CloseRequestFcn = @(~, ~) cleanupAndClose();

    % ============================ Fonctions internes ========================
    function setKeyState(key, isPressed)
        switch key
            case "uparrow",    keysHeld.up    = isPressed;
            case "downarrow",  keysHeld.down  = isPressed;
            case "leftarrow",  keysHeld.left  = isPressed;
            case "rightarrow", keysHeld.right = isPressed;
            case "r"
                if isPressed
                    resetSimulation();
                end
        end
    end

    function updateStep()
        % --- Rampe accelerateur (comme une pedale relachee progressivement) ---
        if keysHeld.up
            pedalState.throttle = min(100, pedalState.throttle + RAMP_THROTTLE_UP * dt);
        else
            pedalState.throttle = max(0, pedalState.throttle - RAMP_THROTTLE_DOWN * dt);
        end

        % --- Rampe frein ---
        if keysHeld.down
            pedalState.brake = min(100, pedalState.brake + RAMP_BRAKE_UP * dt);
        else
            pedalState.brake = max(0, pedalState.brake - RAMP_BRAKE_DOWN * dt);
        end

        % --- Rampe volant (auto-centrage au relachement) ---
        if keysHeld.left && ~keysHeld.right
            pedalState.steering = max(-STEER_MAX, pedalState.steering - RAMP_STEER * dt);
        elseif keysHeld.right && ~keysHeld.left
            pedalState.steering = min(STEER_MAX, pedalState.steering + RAMP_STEER * dt);
        else
            % Ni gauche ni droite (ou les deux) : retour au centre
            if pedalState.steering > 0
                pedalState.steering = max(0, pedalState.steering - RAMP_STEER_CENTER * dt);
            elseif pedalState.steering < 0
                pedalState.steering = min(0, pedalState.steering + RAMP_STEER_CENTER * dt);
            end
        end

        % --- Mise a jour des jauges visuelles ---
        sldThrottle.Value = pedalState.throttle;
        sldBrake.Value    = pedalState.brake;
        sldSteer.Value    = pedalState.steering;
        lblThrottle.Text = sprintf("Accelerateur : %.0f %%", pedalState.throttle);
        lblBrake.Text    = sprintf("Frein : %.0f %%", pedalState.brake);
        lblSteer.Text    = sprintf("Angle volant : %.0f deg", pedalState.steering);

        % --- Pas de simulation vehicule ---
        throttleFrac = pedalState.throttle / 100;
        brakeFrac    = pedalState.brake / 100;
        steeringRad  = deg2rad(pedalState.steering);

        cmd = DriverCommand(throttleFrac, brakeFrac, steeringRad, 0);
        state = vd.step(cmd.toStruct(), dt);
        cleanSensorSignals = sensors.step(state);
        sensorSignals = errorInjector.step(cleanSensorSignals, state);
        ecuMeasurements = ecu.step(sensorSignals);
        appendSensorSignals(sensorSignals);

        wheelSpeedKmh = state.wheelSpeedTrue * 3.6;  % FL, FR, RL, RR
        updateErrorRange(wheelSpeedKmh, ecuMeasurements);

        lblVehicleSpeed.Text = sprintf("Vitesse vehicule : %.1f km/h", state.vehicleSpeed * 3.6);
        lblFL.Text = wheelLabel("FL", wheelSpeedKmh(1), sensorSignals.FL.level, ecuMeasurements, 1);
        lblFR.Text = wheelLabel("FR", wheelSpeedKmh(2), sensorSignals.FR.level, ecuMeasurements, 2);
        lblRL.Text = wheelLabel("RL", wheelSpeedKmh(3), sensorSignals.RL.level, ecuMeasurements, 3);
        lblRR.Text = wheelLabel("RR", wheelSpeedKmh(4), sensorSignals.RR.level, ecuMeasurements, 4);
        updateSignalPlot(state.time, sensorSignals);
    end

    function resetSimulation()
        keysHeld = struct("up", false, "down", false, "left", false, "right", false);
        pedalState = struct("throttle", 0, "brake", 0, "steering", 0);

        sldThrottle.Value = 0;
        sldBrake.Value = 0;
        sldSteer.Value = 0;
        lblThrottle.Text = "Accelerateur : 0 %";
        lblBrake.Text = "Frein : 0 %";
        lblSteer.Text = "Angle volant : 0 deg";

        vd.initialize(DriverCommand(0, 0, 0, 0).toStruct());
        sensors.reset();
        errorInjector.reset();
        ecu.reset();
        signalHistory = emptySignalHistory();
        minErrorKmh = Inf;
        maxErrorKmh = -Inf;
        cla(signalAxes);
        signalAxes.YLim = [-0.2 1.2];
        signalAxes.YTick = [0 1];
        lblSignal.Text = sprintf("Signal brut : %s (0 fronts)", selectedSignalWheel);
        lblErrorRange.Text = "Ecart ECU - reel : aucune mesure valide";
    end

    function history = emptySignalHistory()
        history = struct();
        for wheel = wheelNames
            history.(wheel) = struct("time", 0, "level", false);
        end
    end

    function appendSensorSignals(sensorSignals)
        for wheel = wheelNames
            raw = sensorSignals.(wheel);
            history = signalHistory.(wheel);
            history.time = [history.time, raw.edgeTimes];
            history.level = [history.level, raw.edgeLevels];
            signalHistory.(wheel) = history;
        end
    end

    function updateSignalPlot(time, sensorSignals)
        raw = sensorSignals.(selectedSignalWheel);
        history = signalHistory.(selectedSignalWheel);
        windowStart = max(0, time - SIGNAL_WINDOW);

        % Conserver l'etat juste avant la fenetre pour dessiner un signal
        % continu, puis tous les fronts contenus dans la fenetre.
        previousIndex = find(history.time <= windowStart, 1, "last");
        visibleIndices = find(history.time > windowStart);
        plotTimes = [windowStart, history.time(visibleIndices), time];
        plotLevels = [history.level(previousIndex), history.level(visibleIndices), raw.level];

        cla(signalAxes);
        stairs(signalAxes, plotTimes, double(plotLevels), "LineWidth", 1.4);
        signalAxes.XLim = [windowStart, max(windowStart + SIGNAL_WINDOW, time)];
        signalAxes.YLim = [-0.2 1.2];
        signalAxes.YTick = [0 1];
        grid(signalAxes, "on");
        lblSignal.Text = sprintf("Signal brut : %s (%d fronts cumules)", ...
            selectedSignalWheel, raw.totalEdges);
    end

    function selectSignalWheel(wheel)
        selectedSignalWheel = wheel;
    end

    function updateErrorRange(realSpeedKmh, measurements)
        validIndices = measurements.valid & isfinite(measurements.wheelSpeedMeasured);
        if ~any(validIndices)
            return;
        end

        ecuSpeedKmh = measurements.wheelSpeedMeasured(validIndices) * 3.6;
        signedErrors = ecuSpeedKmh - realSpeedKmh(validIndices);
        minErrorKmh = min(minErrorKmh, min(signedErrors));
        maxErrorKmh = max(maxErrorKmh, max(signedErrors));
        lblErrorRange.Text = sprintf( ...
            "Ecart VECU - Vreelle : min=%.4f km/h | max=%.4f km/h", ...
            minErrorKmh, maxErrorKmh);
    end

    function text = wheelLabel(wheel, speedKmh, level, measurements, index)
        if measurements.valid(index)
            ecuSpeedKmh = measurements.wheelSpeedMeasured(index) * 3.6;
            ecuText = sprintf("ECU=%.1f", ecuSpeedKmh);
        else
            ecuText = "ECU=--";
        end
        text = sprintf("%s : %.1f km/h | %s | S=%d", ...
            wheel, speedKmh, ecuText, level);
    end

    function cleanupAndClose()
        stop(t);
        delete(t);
        delete(fig);
    end
end
