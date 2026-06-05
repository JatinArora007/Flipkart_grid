import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def process_features(df):
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_in_mins'] = df['hour'] * 60 + df['minute']
    
    # Cyclic time features
    df['sin_time'] = np.sin(2 * np.pi * df['time_in_mins'] / (24 * 60))
    df['cos_time'] = np.cos(2 * np.pi * df['time_in_mins'] / (24 * 60))
    
    # Day features
    df['day_of_week'] = df['day'] % 7
    
    # Geohash features
    df['geohash_1'] = df['geohash'].str[:1]
    df['geohash_2'] = df['geohash'].str[:2]
    df['geohash_3'] = df['geohash'].str[:3]
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    
    return df

print("Loading data...")
train_df = pd.read_csv('train.csv')

print("Processing time...")
train_df = process_features(train_df)

y = train_df['demand']
features = [c for c in train_df.columns if c not in ['Index', 'demand', 'timestamp']]

# Train/val split before target encoding to avoid leakage!
train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)

df_train_part = train_df.iloc[train_idx].copy()
df_val_part = train_df.iloc[val_idx].copy()

# Target encoding features
te_cols = ['geohash', 'RoadType', 'Weather', 'Landmarks', 'LargeVehicles']
for col in te_cols:
    means = df_train_part.groupby(col)['demand'].mean()
    global_mean = df_train_part['demand'].mean()
    df_train_part[col + '_te'] = df_train_part[col].map(means).fillna(global_mean)
    df_val_part[col + '_te'] = df_val_part[col].map(means).fillna(global_mean)

df_all = pd.concat([df_train_part, df_val_part], axis=0)

cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 
            'geohash_1', 'geohash_2', 'geohash_3', 'geohash_4', 'geohash_5']

for col in cat_cols:
    df_all[col] = df_all[col].fillna('Unknown')
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

num_cols = ['day', 'NumberofLanes', 'Temperature', 'hour', 'minute', 'time_in_mins', 'sin_time', 'cos_time', 'day_of_week']
for col in num_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# Re-split based on lengths
X_train = df_all.iloc[:len(train_idx)][[c for c in df_all.columns if c not in ['Index', 'demand', 'timestamp']]]
y_train = df_all.iloc[:len(train_idx)]['demand']

X_val = df_all.iloc[len(train_idx):][[c for c in df_all.columns if c not in ['Index', 'demand', 'timestamp']]]
y_val = df_all.iloc[len(train_idx):]['demand']

print("Training model...")
model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=8, random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train)

print("Predicting...")
preds = model.predict(X_val)

r2 = r2_score(y_val, preds)
score = max(0, 100 * r2)

print(f"Validation R2 Score: {r2}")
print(f"Competition Score: {score}")
