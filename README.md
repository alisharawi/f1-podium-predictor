# Formula 1 Podium Prediction Pipeline

An end-to-end machine learning pipeline that predicts Formula 1 Grand Prix podium finishers using LightGBM, leakage-free rolling performance features, and automated post-race retraining powered by the FastF1 API.

## Overview

Formula 1 race outcomes depend on single-lap qualifying pace, starting grid penalties, and recent driver and constructor form. This project transforms historical Grand Prix data into a time-aware tabular feature set, trains a LightGBM binary classifier to estimate podium probabilities, and automatically ingests new 2026 race weekend results via FastF1 to keep both the dataset and trained model up to date.

## Key Features

- **Leakage-Free Feature Engineering:** Computes 3-race rolling average points for drivers and constructors using `shift(1)` so current-race results never leak into training features.
- **Temporal Validation:** Trains on pre-2025 seasons and evaluates strictly out-of-sample on the 2025 season using ROC AUC and average correct podium finishers per race.
- **Pre-Race Inference Engine:** Takes the official starting grid for an upcoming Grand Prix (including grid penalties), maps each driver's and constructor's latest rolling form, and outputs the projected winner, projected Top-3 podium, and full-grid podium probabilities.
- **Automated Post-Race Retraining:** Uses FastF1 with local session caching to pull official Qualifying and Race results after each round, append newly engineered rows to `f1_engineered_features.csv`, and overwrite `f1_podium_model.pkl` with the retrained classifier.

## Project Structure

```text
f1-podium-predictor/
├── feature_engineering_code.py   # Cleans raw data and engineers rolling & qualifying delta features
├── model_train.py                # Trains LightGBM on temporal split, evaluates, and exports .pkl model
├── predict_race.py               # Runs pre-race podium predictions on an upcoming starting grid
├── update_model.py               # Pulls new race data via FastF1, updates CSV, and retrains model
├── f1_training_dataset.csv       # Raw historical F1 race and qualifying dataset (158 KB)
├── f1_engineered_features.csv    # Processed dataset with engineered features, auto-updated (218 KB)
├── f1_podium_model.pkl           # Serialized LightGBM classifier artifact (99.5 KB)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes f1_cache/, fastf1_cache/, and Python artifacts
└── README.md                     # Project documentation
