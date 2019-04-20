import matplotlib.pyplot as plt, numpy as np
from pandas import read_csv
from sklearn import linear_model
#np.set_printoptions(threshold=np.inf)
from sklearn.preprocessing import PolynomialFeatures
import math
from sklearn.metrics import mean_squared_error


data = None
with open('BLK2.csv') as csv_file:
    data = read_csv(csv_file)

#print(data.corr())

#header = [
#    'open', 
#    'high',
#    'low',
#    'adjclose',
#    'volume' 
#]

header = [ 'adjclose' ]

def fit(data):
    X = data[header].as_matrix()
    y = data['close'].as_matrix()

    poly_reg= PolynomialFeatures(degree=4)
    x_poly=poly_reg.fit_transform(X)
    poly_reg.fit(x_poly, y)
    lin_reg2=linear_model.LinearRegression()
    lin_reg2.fit(x_poly, y)
    

    #Предсказывание
    print((lin_reg2.predict(poly_reg.fit_transform(X))))
    #print(len(lin_reg2.predict(poly_reg.fit_transform(X)))) 
    rmse_test = np.sqrt(mean_squared_error(y, (lin_reg2.predict(poly_reg.fit_transform(X)))))
    print('MSE : ',rmse_test)

fit(data)
