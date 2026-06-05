import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

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
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        
    return df

if __name__ == '__main__':
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Save test index before dropping
    test_idx = test_df['Index'].copy()

    print("Processing features...")
    train_df = process_features(train_df)
    test_df = process_features(test_df)
    
    if 'Index' in train_df.columns:
        train_df = train_df.drop(columns=['Index'])
    if 'Index' in test_df.columns:
        test_df = test_df.drop(columns=['Index'])

    print("Training AutoGluon model on full dataset (this may take a while)...")
    # Using 'best_quality' to perform stacking/bagging
    predictor = TabularPredictor(label='demand', eval_metric='r2', problem_type='regression').fit(
        train_df,
        presets='best_quality',
        time_limit=1200  # Limited to 20 minutes to prevent running for hours
    )

    print("Predicting...")
    preds = predictor.predict(test_df)
    preds = np.clip(preds, 0, None)

    sub = pd.DataFrame()
    sub['Index'] = test_idx
    sub['demand'] = preds
    sub.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
