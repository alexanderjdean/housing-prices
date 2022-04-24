from util import get_and_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = get_and_split()
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
print("The R2 score for random forest regression is " + str(r2_score(y_test, y_pred)))