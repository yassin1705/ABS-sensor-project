% GENERATE_FAULTY_BRAKING_DATASET_5000
% Genere 5 000 scenarios avec une perte intermittente reelle de fronts ABS.
%
% La campagne conserve les parametres du dataset d'apprentissage :
%   - duree de 5 s et frequence ECU de 100 Hz ;
%   - repartition 30/30/30/10 des phenomenes physiques ;
%   - vitesse initiale stratifiee entre 40 et 100 km/h ;
%   - freinage et braquage non nuls ;
%   - latence et jitter nominaux healthy_realistic.
%
% Chaque scenario contient un defaut sur une seule roue. Les roues et les
% severites sont equilibrees de facon deterministe. La suppression est faite
% au niveau des fronts par SensorErrorInjector, avant le calcul de vitesse ECU.

projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot, "core"));

numberOfSimulations = 5000;
durationSeconds = 5.0;
campaignSeed = 40000;
outputFolder = fullfile(projectRoot, "simulation_results");

generator = BrakingDatasetGenerator( ...
    "", outputFolder, campaignSeed);

[datasetPath, manifestPath, manifest] = generator.generate( ...
    numberOfSimulations, durationSeconds, ...
    InjectIntermittentLoss=true, ...
    FaultStartRangeSeconds=[1.5, 2.0], ...
    FaultDurationRangeSeconds=[1.5, 2.0], ...
    FaultSeverityLevels=[0.25, 0.50, 0.75, 1.00], ...
    FaultDropoutProbability=0.50, ...
    ClassifyObserved=true, ...
    ChunkSize=25, ...
    ProgressEvery=100);

fprintf("\nCampagne ABS avec defauts terminee.\n");
fprintf("Scenarios : %d\n", height(manifest));
fprintf("Dataset   : %s\n", datasetPath);
fprintf("Manifeste : %s\n", manifestPath);
