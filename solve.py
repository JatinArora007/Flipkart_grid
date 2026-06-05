import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

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
test_df = pd.read_csv('test.csv')

print("Processing features...")
train_df = process_features(train_df)
test_df = process_features(test_df)

y = train_df['demand']

print("Applying Target Encoding...")
# KFold Target Encoding for train to prevent overfitting
te_cols = ['geohash', 'RoadType', 'Weather', 'Landmarks', 'LargeVehicles']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in te_cols:
    train_df[col + '_te'] = np.nan
    
    # Calculate for test data using ALL train data
    global_mean = train_df['demand'].mean()
    means = train_df.groupby(col)['demand'].mean()
    test_df[col + '_te'] = test_df[col].map(means).fillna(global_mean)
    
    # Calculate for train data using out-of-fold
    for tr_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        fold_means = X_tr.groupby(col)['demand'].mean()
        train_df.loc[val_idx, col + '_te'] = train_df.loc[val_idx, col].map(fold_means).fillna(global_mean)

features = [c for c in train_df.columns if c not in ['Index', 'demand', 'timestamp']]

df_all = pd.concat([train_df[features], test_df[features]], axis=0)

cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 
            'geohash_1', 'geohash_2', 'geohash_3', 'geohash_4', 'geohash_5']

for col in cat_cols:
    df_all[col] = df_all[col].fillna('Unknown')
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

num_cols = ['day', 'NumberofLanes', 'Temperature', 'hour', 'minute', 'time_in_mins', 'sin_time', 'cos_time', 'day_of_week']
for col in num_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

X_train = df_all.iloc[:len(train_df)]
X_test = df_all.iloc[len(train_df):]

print("Training model...")
model = XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=8, random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y)

print("Predicting...")
preds = model.predict(X_test)

sub = pd.DataFrame()
sub['Index'] = test_df['Index']
sub['demand'] = preds
# Make sure no negative demand predictions if demand >= 0
sub['demand'] = sub['demand'].clip(lower=0) 
sub.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
