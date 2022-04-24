from util import get_and_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = get_and_split()
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("The R2 score for decision tree regression is " + str(r2_score(y_test, y_pred)))