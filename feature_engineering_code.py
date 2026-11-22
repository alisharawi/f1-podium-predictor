import pandas as pd
import numpy as np

df = pd.read_csv('f1_training_dataset.csv')

# Ensure correct chronological order
df['EventDate'] = pd.to_datetime(df['EventDate'])
df = df.sort_values(by=['EventDate', 'RoundNumber', 'GridPosition']).reset_index(drop=True)

# Clean numeric values
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
df['QualiPosition'] = pd.to_numeric(df['QualiPosition'], errors='coerce')
df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)

# Replace missing grid positions (e.g. pit-lane starts) with 20
df['GridPosition'] = df['GridPosition'].fillna(20)

# ---------------------------------------------------------
# 1. Driver Momentum: Rolling average points over last 3 races
# ---------------------------------------------------------
# We use shift(1) so the current race's points are NOT leaked into the feature!
df['driver_pts_last3'] = (
    df.groupby('Abbreviation')['Points']
    .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .reset_index(level=0, drop=True)
)

# ---------------------------------------------------------
# 2. Constructor Momentum: Rolling average points per team
# ---------------------------------------------------------
team_race_points = df.groupby(['Season', 'RoundNumber', 'TeamName'])['Points'].sum().reset_index()
team_race_points = team_race_points.sort_values(['Season', 'RoundNumber'])
team_race_points['team_pts_last3'] = (
    team_race_points.groupby('TeamName')['Points']
    .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .reset_index(level=0, drop=True)
)

df = pd.merge(df, team_race_points[['Season', 'RoundNumber', 'TeamName', 'team_pts_last3']], 
              on=['Season', 'RoundNumber', 'TeamName'], how='left')

# ---------------------------------------------------------
# 3. Teammate Head-to-Head Delta in Qualifying
# ---------------------------------------------------------
# A lower quali rank than teammate indicates driver pace superiority
team_avg_quali = df.groupby(['Season', 'RoundNumber', 'TeamName'])['QualiPosition'].transform('mean')
df['teammate_quali_delta'] = df['QualiPosition'] - team_avg_quali

# ---------------------------------------------------------
# 4. Fill initial early-season NaN rolling stats with zeros
# ---------------------------------------------------------
df['driver_pts_last3'] = df['driver_pts_last3'].fillna(0)
df['team_pts_last3'] = df['team_pts_last3'].fillna(0)
df['teammate_quali_delta'] = df['teammate_quali_delta'].fillna(0)

# Save processed dataset
df.to_csv('f1_engineered_features.csv', index=False)
print("Engineered dataset saved successfully!")