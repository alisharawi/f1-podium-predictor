import pandas as pd
import lightgbm as lgb
import joblib
import fastf1
import os

# ==========================================
# 1. CONFIGURATION: CHANGE THESE AFTER EVERY RACE
# ==========================================
SEASON = 2026
ROUND_NUMBER = 19  # e.g., 20 for Mexico, 21 for Brazil, etc.

# Enable FastF1 cache to avoid re-downloading data
# Create a folder named 'fastf1_cache' in your project directory
if not os.path.exists('fastf1_cache'):
    os.makedirs('fastf1_cache')
fastf1.Cache.enable_cache('fastf1_cache')

print(f"Fetching data for Season {SEASON}, Round {ROUND_NUMBER}...")

# ==========================================
# 2. AUTOMATICALLY FETCH RACE AND QUALI DATA
# ==========================================
# Load Race and Qualifying sessions
race = fastf1.get_session(SEASON, ROUND_NUMBER, 'R')
race.load(weather=False, telemetry=False, messages=False)

quali = fastf1.get_session(SEASON, ROUND_NUMBER, 'Q')
quali.load(weather=False, telemetry=False, messages=False)

# Extract official final Race results and Grid positions
race_results = race.results[['Abbreviation', 'TeamName', 'GridPosition', 'Position']].copy()

# Extract official Qualifying positions
quali_results = quali.results[['Abbreviation', 'Position']].copy()
quali_results = quali_results.rename(columns={'Position': 'QualiPosition'})

# Merge into a single dataframe for this race
update_df = pd.merge(race_results, quali_results, on='Abbreviation', how='left')

# Format the target variables and metadata
update_df['Position'] = update_df['Position'].fillna(20) # Handle DNFs
update_df['is_podium'] = update_df['Position'].apply(lambda x: 1 if x <= 3 else 0)
update_df['Season'] = SEASON
update_df['RoundNumber'] = ROUND_NUMBER
update_df['EventDate'] = race.date.strftime('%Y-%m-%d')

# ==========================================
# 3. ENGINEER FEATURES FROM HISTORICAL DATA
# ==========================================
historical_df = pd.read_csv('f1_engineered_features.csv')

# Grab the latest form prior to this race
latest_driver_form = historical_df.sort_values('EventDate').groupby('Abbreviation')['driver_pts_last3'].last().to_dict()
latest_team_form = historical_df.sort_values('EventDate').groupby('TeamName')['team_pts_last3'].last().to_dict()

# Map the form to the current race
update_df['driver_pts_last3'] = update_df['Abbreviation'].map(latest_driver_form).fillna(0)
update_df['team_pts_last3'] = update_df['TeamName'].map(latest_team_form).fillna(0)

# Calculate teammate delta dynamically
update_df['teammate_quali_delta'] = update_df['QualiPosition'] - update_df.groupby('TeamName')['QualiPosition'].transform('mean')

# ==========================================
# 4. APPEND TO DATASET AND SAVE
# ==========================================
updated_historical = pd.concat([historical_df, update_df], ignore_index=True)
updated_historical.to_csv('f1_engineered_features.csv', index=False)
print(f"✅ Round {ROUND_NUMBER} appended to f1_engineered_features.csv")

# ==========================================
# 5. RETRAIN AND OVERWRITE MODEL
# ==========================================
feature_cols = ['GridPosition', 'QualiPosition', 'driver_pts_last3', 'team_pts_last3', 'teammate_quali_delta']
X_train = updated_historical[feature_cols]
y_train = updated_historical['is_podium']

clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

joblib.dump(clf, 'f1_podium_model.pkl')
print(f"✅ Model retrained with Round {ROUND_NUMBER} data and saved. Ready for the next race!")