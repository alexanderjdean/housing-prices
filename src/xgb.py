from xgboost import XGBRegressor
from util import get_and_split, generate_submission
import pandas as pd
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = get_and_split()
regressor = XGBRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("The R2 score for XGBoost regression is " + str(r2_score(y_test, y_pred)))

print("Generating real submission for competition...")
test = pd.read_csv('data/processed_test.csv')
test = test.iloc[:, :].values

prediction = regressor.predict(test)
generate_submission(prediction)