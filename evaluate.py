import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def process_features(df):
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_in_mins'] = df['hour'] * 60 + df['minute']
    
    # Cyclic time features
    df['sin_time'] = np.sin(2 * np.pi * df['time_in_mins'] / (24 * 60))
    df['cos_time'] = np.cos(2 * np.pi * df['time_in_mins'] / (24 * 60))
    
    # Day features
    df['day_of_week'] = df['day'] % 7
    
    # Geohash features (helps Tree models generalize spatial regions)
    df['geohash_1'] = df['geohash'].str[:1]
    df['geohash_2'] = df['geohash'].str[:2]
    df['geohash_3'] = df['geohash'].str[:3]
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    
    # Drop useless columns for AutoGluon
    if 'Index' in df.columns:
        df = df.drop(columns=['Index'])
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        
    return df

if __name__ == '__main__':
    print("Loading data...")
    train_df = pd.read_csv('train.csv')

    print("Processing features...")
    train_df = process_features(train_df)

    # Train/val split
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    print("Training AutoGluon model (this may take a while)...")
    # Using 'best_quality' to get the absolute best score via bagging/stacking
    predictor = TabularPredictor(label='demand', eval_metric='r2', problem_type='regression').fit(
        train_data,
        presets='best_quality',
        time_limit=1200  # Limiting to 20 minutes max for evaluation
    )

    print("Predicting...")
    preds = predictor.predict(val_data)
    preds = np.clip(preds, 0, None)

    r2 = r2_score(val_data['demand'], preds)
    score = max(0, 100 * r2)

    print(f"Validation R2 Score: {r2}")
    print(f"Competition Score: {score}")
    print("\nAutoGluon Leaderboard:")
    print(predictor.leaderboard(val_data))
