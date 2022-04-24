import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

CATEGORICAL_INDICES = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 
                       21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 34, 
                       38, 39, 40, 41, 52, 54, 56, 57, 59, 62, 63, 64, 71, 
                       72, 73, 77, 78]
ROW_TEST_INDEX_START = 1460
NAN_INDICES = [2, 25, 58]

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.drop('Id', inplace=True, axis=1)
test.drop('Id', inplace=True, axis=1)

train = train.iloc[:, :].values
test = test.iloc[:, :].values

labels = train[:, -1]
labels = labels.reshape(labels.shape[0], 1)
train = train[:,:-1]
combined = np.concatenate((train, test))

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

for index in NAN_INDICES:
    imputer.fit(combined[:, index].reshape(-1, 1))
    combined[:, index] = imputer.transform(combined[:, index].reshape(-1, 1)).reshape(-1)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), CATEGORICAL_INDICES)], 
                     remainder='passthrough')

combined = np.array(ct.fit_transform(combined))
train = combined[:ROW_TEST_INDEX_START, :]
test = combined[ROW_TEST_INDEX_START:, :]
train = np.append(train, labels, axis=1)

pd.DataFrame(train).to_csv("./data/processed_train.csv")
pd.DataFrame(test).to_csv("./data/processed_test.csv")
