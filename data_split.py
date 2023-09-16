import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('./dataset/ECG_adult_age_train.csv')
    files = df['FILENAME'].to_list()
    ages  = df['AGE'].to_list()
    ratio = [0.2, 0.25, 0.33, 0.5]
    folds = {}
    for idx, r in enumerate(ratio):
        X_train, X_test, y_train, y_test = train_test_split(files, ages, test_size=r, random_state=42)
        
        folds[f'{idx+1}'] = [X_test, y_test]
        files = X_train
        ages  = y_train
        print(f'FOLD {idx+1} COUNT : {len(X_test)}')
    
    idx += 1
    folds[f'{idx+1}'] = [X_train, y_train]
    print(f'FOLD {idx+1} COUNT : {len(X_train)}')
    