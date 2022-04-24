import pandas as pd
from sklearn.model_selection import train_test_split

def get_and_split():
    dataset = pd.read_csv('data/processed_train.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test