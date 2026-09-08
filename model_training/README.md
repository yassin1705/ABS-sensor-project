# Entraînement du prédicteur de vitesses ABS

Ce dossier contient un pipeline commun pour trois architectures :

- `CNNForecaster`
- `GRUForecaster`
- `LSTMForecaster`

Toutes respectent le contrat :

```text
[batch, 20 pas historiques, 4 roues]
    -> [batch, 5 pas futurs, 4 roues]
```

L’architecture est définie directement dans chaque classe de modèle. Il n’y a
pas de classe de configuration.

## Installation

Python 3.11 ou 3.12 est recommandé.

```powershell
cd .\model_training
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Notebook

`notebooks/train_models.ipynb` exécute tout le pipeline :

1. sélection du CSV MATLAB et du manifeste ;
2. préparation ou réutilisation du cache HDF5 ;
3. chargement des partitions ;
4. entraînement CNN, GRU et LSTM avec le même `Trainer` ;
5. comparaison des métriques.

Le CSV MATLAB reste la source officielle. Le cache HDF5 évite de relire
plusieurs gigaoctets de texte à chaque époque.

## Monitoring du test

Les métriques test sont affichées à chaque époque, conformément au besoin du
projet. Elles sont strictement informatives :

- l’early stopping utilise la validation ;
- le meilleur checkpoint utilise la validation ;
- le scheduler utilise la validation ;
- aucune décision automatique ne dépend du test.

Cette séparation évite que le monitoring du test ne transforme indirectement
le test en partition de validation.

## Calibrage MSP des résidus GRU

Le module `spc_calibration.py` construit quatre cartes de contrôle à partir
des résidus sains `mesure - prédiction` au premier horizon du GRU. Il lit le
CSV MATLAB par blocs, sélectionne des simulations complètes de validation et
utilise un pas de fenêtre de 10 ms, identique au fonctionnement en ligne.

Exemple POC sur 100 simulations :

```powershell
Set-Location <repository-root>
.\model_training\.venv\Scripts\python.exe -m model_training.spc_calibration `
    --simulation-count 100 `
    --selection random `
    --selection-seed 42
```

Les sorties sont écrites dans `model_training/spc_calibration` :

- `control_limits.json` : moyenne, écart-type, LCL et UCL par roue ;
- `healthy_t_plus_one_residuals.csv` : mesures, prédictions et résidus ;
- `healthy_control_charts.png` : aperçu des quatre cartes.

Une fenêtre est ignorée si une des 20 mesures historiques ou la cible `t+1`
est invalide. Les mesures invalides devront être surveillées séparément comme
indicateur de qualité du signal.

## Rejeu temps réel des cartes MSP

Le rejeu lit une simulation complète, alimente causalement le GRU à 100 Hz et
compare chaque prédiction `t+1` avec la mesure suivante. Par défaut, il choisit
une simulation de test saine avec une vitesse variable, qui n'a pas servi au
calibrage :

```powershell
Set-Location <repository-root>
.\model_training\.venv\Scripts\python.exe `
    -m model_training.realtime_spc_replay
```

Sous Windows, il suffit aussi de double-cliquer sur `start_live_spc.cmd` à la
racine du projet. La fenêtre démarre directement le flux à 100 Hz.

Options utiles :

```powershell
# Choisir une simulation et accélérer le rejeu cinq fois
.\model_training\.venv\Scripts\python.exe `
    -m model_training.realtime_spc_replay `
    --simulation-id 2 `
    --playback-speed 5

# Vérification sans fenêtre graphique et sans attente
.\model_training\.venv\Scripts\python.exe `
    -m model_training.realtime_spc_replay `
    --no-live `
    --playback-speed 0
```

Un dépassement individuel produit un avertissement. Une alarme est confirmée
si au moins trois des cinq derniers résidus valides dépassent les limites.
Les valeurs ECU invalides sont affichées séparément et ne produisent pas de
résidu numérique. L'échelle verticale reste centrée sur les limites de
contrôle afin qu'un pic extrême ne rende pas le reste du signal visuellement
plat. Les valeurs hors échelle restent enregistrées sans modification et sont
affichées sur le bord du graphique.

Pour rejouer le dernier dataset contenant les pertes intermittentes injectées,
double-cliquer sur `start_faulty_live_spc.cmd` à la racine du projet :

```powershell
.\start_faulty_live_spc.cmd

# Choisir une autre simulation du CSV fautif
.\start_faulty_live_spc.cmd --simulation-id 25
```

Le rejeu fautif utilise exactement le même modèle et le même fichier de
limites saines. Les labels de défaut servent uniquement à colorer l'intervalle
injecté et à calculer le délai de détection après le rejeu ; ils ne sont jamais
fournis au GRU ni à la règle d'alarme.

Comme le GRU est multivarié, une roue corrompue peut provoquer des dépassements
résiduels sur les quatre sorties. Le tableau de bord affiche ces effets croisés
en orange. Le capteur suspecté en rouge est localisé séparément à partir d'une
rupture locale de vitesse, comparée au changement médian des quatre roues. Une
localisation exige deux votes dans les cinq derniers échantillons.

## Second modèle : estimation des paramètres du défaut

`fault_parameter_estimation.py` apprend sur la série complète de 5 s du seul
capteur fautif. L'entrée contient la vitesse ECU et son indicateur de validité,
soit `[batch, 500, 2]`. La sortie contient trois valeurs normalisées :

- instant de début du défaut ;
- durée du défaut, dont est déduite la fin ;
- sévérité injectée.

Ce modèle intervient après la détection/localisation MSP : on lui transmet la
série du capteur suspecté. Une série saine produirait malgré tout trois valeurs,
car ce second modèle est un estimateur de paramètres et non un détecteur.

Le type `intermittent_loss` et la probabilité de base `0.50` sont constants
dans cette campagne. Ils sont donc conservés dans les métadonnées au lieu
d'être artificiellement appris. La probabilité effective est déduite par
`sévérité * 0.50`. Les colonnes de vérité physique, `fault_active` et
`dropped_edges`, ne sont jamais utilisées comme entrées du modèle.

Pour préparer les scénarios dans lesquels au moins un front a vraiment été
supprimé, puis entraîner le modèle :

```powershell
Set-Location <repository-root>
.\train_fault_parameter_model.cmd
```

Vérification rapide sur un sous-ensemble, sans lancer l'entraînement complet :

```powershell
.\train_fault_parameter_model.cmd --max-simulations 64 --epochs 2 `
    --output-directory model_training\experiments\fault_parameter_gru_smoke `
    --cache model_training\cache\fault_parameter_smoke.npz
```

Les sorties comprennent `best_model.pt`, `fault_parameter_gru.onnx`,
`metadata.json`, `metrics.json` et l'historique d'entraînement. Les partitions
sont faites par simulation et stratifiées par régime physique et sévérité.

Le benchmark complet CNN/GRU/LSTM de ce second problème est maintenant isolé
dans `fault_parameter_training/notebooks/train_models.ipynb`. Il mélange les
simulations saines et fautives et ajoute une sortie de probabilité de défaut
avant les paramètres conditionnels. Le script `start_fault_parameter_notebook.cmd`
ouvre directement ce notebook.
