import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('f1_engineered_features.csv')

# Drop any unclassified or DNFs without positions if predicting finishes
df = df.dropna(subset=['Position'])

# Temporal Split: Past seasons for training, latest season for evaluation
train_df = df[df['Season'] < 2025]
test_df = df[df['Season'] == 2025].copy()

feature_cols = [
    'GridPosition',
    'QualiPosition',
    'driver_pts_last3',
    'team_pts_last3',
    'teammate_quali_delta'
]

X_train = train_df[feature_cols]
y_train = train_df['is_podium']

X_test = test_df[feature_cols]
y_test = test_df['is_podium']

# Train LightGBM
clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
test_df['podium_prob'] = clf.predict_proba(X_test)[:, 1]
print(f"ROC AUC Score: {roc_auc_score(y_test, test_df['podium_prob']):.3f}")

hits = []
for (season, rnd), race in test_df.groupby(['Season', 'RoundNumber']):
    actual_top3 = set(race[race['Position'] <= 3]['Abbreviation'])
    predicted_top3 = set(race.sort_values(by='podium_prob', ascending=False).head(3)['Abbreviation'])
    hits.append(len(actual_top3.intersection(predicted_top3)))

print(f"\nAverage correct podium finishers per race: {sum(hits) / len(hits):.2f} / 3")

# --- STEP 2: SAVE THE TRAINED MODEL FOR FUTURE RACES ---
import joblib
joblib.dump(clf, 'f1_podium_model.pkl')
print("Model saved to 'f1_podium_model.pkl'")