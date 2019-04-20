import numpy as np
import matplotlib.pyplot as plt, numpy as np
from pandas import read_csv
import csv
#np.set_printoptions(threshold=np.inf)
from sklearn.metrics import mean_squared_error


class Polynomial_regression: 

    def __init__(self, weights=None):
        self.weights = np.array(weights) if weights is not None else None

    @property
    def order(self):
        return len(self.weights) if self.weights is not None else 0

    def evaluate(self, x):
        return np.dot(np.vander(x, self.order), self.weights[:, np.newaxis]).ravel()

    def fit(self, X, y=None):
        self.weights = (np.linalg.pinv(np.vander(X, self.order)) @ y[:, np.newaxis]).ravel()

    def predict(self, X):
        if self.weights is not None: 
            return self.evaluate(X)
        else:
            raise Exception("Model wasn't fitted. Fit model first. ")

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)







with open('in.csv', 'r') as f:
  reader = csv.reader(f)
  list = list(reader)

#print(list)

X_list=[]# список всего, кроме результата
y_list=[]# результат

for elem in list:
    X_list.append(float(elem[6]))
    y_list.append(float(elem[7]))

#print('X_list',X_list)
#print('y_list',y_list)

X = np.array(X_list)
y = np.array(y_list)


reg = Polynomial_regression()

reg.weights = np.array([1,2,3])  # 3 + 2x + 1x^2

print(reg.fit_predict(y,X))

rmse_test = np.sqrt(mean_squared_error(X, reg.fit_predict(y,X)))
print('MSE : ',rmse_test)

