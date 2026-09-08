# Estimation des paramètres de défaut ABS

Ce dossier est le second pipeline d'apprentissage du projet. Il conserve la
même organisation que `model_training`, mais traite une série provenant d'un
seul capteur. Il réalise d'abord une classification sain/fautif puis, si le
capteur est fautif, estime :

- le début du défaut ;
- sa durée et donc sa fin ;
- sa sévérité.

Le type `intermittent_loss` et la probabilité de base `0.50` sont constants
dans la campagne actuelle et restent dans les métadonnées. La probabilité
effective est `sévérité × 0.50`.

## Organisation

```text
fault_parameter_training/
├── data.py
├── evaluation.py
├── monitoring.py
├── trainer.py
├── models/
│   ├── base.py
│   ├── cnn.py
│   ├── gru.py
│   └── lstm.py
├── notebooks/train_models.ipynb
├── cache/
├── experiments/
└── tests/
```

## Benchmark

Ouvrir le notebook suivant et exécuter les cellules dans l'ordre :

```powershell
Set-Location <repository-root>
.\start_fault_parameter_notebook.cmd
```

Le raccourci historique `train_fault_parameter_model.cmd` ouvre maintenant le
même notebook afin d'éviter de relancer l'ancien prototype fautif uniquement.

Pour une première vérification, utiliser `MAX_SIMULATIONS = 64` et
`EPOCHS = 2`. Pour le benchmark complet, utiliser `MAX_SIMULATIONS = None` et
`EPOCHS = 50`.

Les courbes train/validation sont enregistrées par modèle et visibles avec :

```powershell
.\model_training\.venv\Scripts\tensorboard.exe `
    --logdir fault_parameter_training\experiments
```

La partition test est évaluée une seule fois après la sélection validation.

CNN, GRU et LSTM possèdent chacun leur propre cellule de chargement et
d'entraînement. Après chaque cellule, le modèle est replacé sur CPU et le cache
CUDA inutilisé est libéré, ce qui évite de conserver plusieurs architectures
simultanément dans la mémoire GPU.

Le benchmark contient aussi un quatrième candidat `CNN–GRU`. Ses convolutions
extraient les ruptures locales dues aux pertes de fronts, puis une GRU
bidirectionnelle encode leur position et leur évolution sur la série complète.
Il possède sa propre cellule, son historique, son checkpoint et peut devenir le
gagnant validation exporté en ONNX comme les autres architectures.

Pour combiner directement les deux modèles déjà entraînés,
`CNNGRUFusionPredictor` charge leurs meilleurs checkpoints. Il moyenne leur
probabilité de défaut, prend le début et la durée du CNN, et la sévérité de la
GRU. Sa méthode `predict(speed_mps, valid)` accepte une série brute de 500
échantillons et retourne le diagnostic ainsi que les paramètres physiques.

## Simulation combinée avec l'ancien MSP

Le lanceur suivant rejoue une simulation fautive dans les deux pipelines :

```powershell
.\start_combined_diagnostic_simulation.cmd
```

Il affiche, pour FL, FR, RL et RR :

- la classe de l'ancien GRU t+1 + MSP (`HEALTHY`, `CROSS_ALARM` ou
  `SUSPECTED_FAULTY`) ;
- la classe et la probabilité de la fusion CNN/GRU ;
- les paramètres physiques estimés lorsque la fusion conclut `FAULTY` ;
- la vérité injectée, uniquement pour évaluer les diagnostics après coup.

Pour une vérification immédiate sans graphique ni délai :

```powershell
.\start_combined_diagnostic_simulation.cmd --no-live --playback-speed 0
```

Les résultats JSON, le CSV MSP et la figure sont écrits dans
`fault_parameter_training/combined_replay`.

## Composition du dataset

Le cache mélange les deux campagnes MATLAB :

- pour chaque simulation fautive observable, la roue injectée est un exemple
  positif et les trois autres roues sont des exemples sains ;
- pour chaque simulation saine, une roue équilibrée est ajoutée comme exemple
  sain supplémentaire ;
- toutes les roues provenant d'une même simulation restent dans la même
  partition afin d'éviter une fuite train/validation/test.

La sortie du modèle est `[probabilité_défaut, début, durée, sévérité]`. La loss
des trois paramètres est masquée pour les capteurs sains. Une BCE pondérée
compense le nombre supérieur d'exemples sains.
