import os
import joblib
import pandas as pd

# ==========================================
# 1. CONFIGURATION (CHANGE THESE EVERY RACE)
# ==========================================
RACE_NAME = "Next Race"

# Official starting grid for the upcoming race
# Format: [Abbreviation, TeamName, GridPosition, QualiPosition]
race_grid = [
    
   ['HAM', 'Ferrari', 1, 1],

    ['RUS', 'Mercedes', 2, 2],

    ['LEC', 'Ferrari', 3, 3],

    ['ANT', 'Mercedes', 4, 4],

    ['VER', 'Red Bull Racing', 5, 4],  # 1-place penalty

    ['NOR', 'McLaren', 6, 6],

    ['HAD', 'Red Bull Racing', 7, 7],

    ['LIN', 'RB', 8, 8],

    ['GAS', 'Alpine', 9, 9],

    ['HUL', 'Kick Sauber', 10, 10],

    ['COL', 'Alpine', 11, 11],

    ['PIA', 'McLaren', 12, 12],

    ['BOR', 'Kick Sauber', 13, 13],

    ['BEA', 'Haas', 14, 14],

    ['SAI', 'Williams', 15, 15],

    ['ALB', 'Williams', 16, 16],

    ['ALO', 'Aston Martin', 17, 17],

    ['HIR', 'Haas', 18, 18],

    ['STR', 'Aston Martin', 19, 19],

    ['PER', 'Cadillac', 20, 20],

    ['BOT', 'Cadillac', 21, 21],  # Back of grid penalty

    ['LAW', 'RB', 22, 5], 
]

# ==========================================
# 2. SYSTEM SETUP (DO NOT CHANGE)
# ==========================================
# Lock paths to your Mawdoo3 Internship folder to prevent Empty DataFrames
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'f1_engineered_features.csv')
model_path = os.path.join(base_dir, 'f1_podium_model.pkl')

clf = joblib.load(model_path)
historical_df = pd.read_csv(csv_path)

# ==========================================
# 3. PREDICTION ENGINE (DO NOT CHANGE)
# ==========================================
# Extract recent form
latest_driver_form = historical_df.sort_values('EventDate').groupby('Abbreviation')['driver_pts_last3'].last().to_dict()
latest_team_form = historical_df.sort_values('EventDate').groupby('TeamName')['team_pts_last3'].last().to_dict()

pred_df = pd.DataFrame(race_grid, columns=['Abbreviation', 'TeamName', 'GridPosition', 'QualiPosition'])

# Map features
pred_df['driver_pts_last3'] = pred_df['Abbreviation'].map(latest_driver_form).fillna(0)
pred_df['team_pts_last3'] = pred_df['TeamName'].map(latest_team_form).fillna(0)
pred_df['teammate_quali_delta'] = pred_df['QualiPosition'] - pred_df.groupby('TeamName')['QualiPosition'].transform('mean')

# Predict
feature_cols = ['GridPosition', 'QualiPosition', 'driver_pts_last3', 'team_pts_last3', 'teammate_quali_delta']
X_live = pred_df[feature_cols]
pred_df['podium_prob'] = clf.predict_proba(X_live)[:, 1]

# Rank
ranked = pred_df.sort_values(by='podium_prob', ascending=False).reset_index(drop=True)

# Output
print("=" * 55)
print(f"       {RACE_NAME.upper()} - MODEL PREDICTIONS       ")
print("=" * 55)
print(f"Projected Winner: {ranked.iloc[0]['Abbreviation']}")
print(f"Projected Podium: {', '.join(ranked.iloc[:3]['Abbreviation'].tolist())}")
print("-" * 55)
print(ranked[['Abbreviation', 'TeamName', 'GridPosition', 'podium_prob']].to_string(index=False))