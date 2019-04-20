import math
import numpy as np
np.set_printoptions(threshold=np.inf)

class LogisticRegression2:
    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, theta=0, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.theta = theta
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        #return 1 / (1 + np.exp(-z)) 
        return .5 * (1 + np.tanh(.5 * z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # инициализируем веса
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print('loss: ', self.__loss(h, y))
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


import sklearn
import sklearn.datasets
import csv

with open('in.csv', 'r') as f:
  reader = csv.reader(f)
  list = list(reader)

#print(list)

X_list=[]# список всего, кроме результата
y_list=[]# результат

for elem in list:
    X_list.append(elem[0:len(elem)-1])
    y_list.append(elem[len(elem)-1])


X = np.array(X_list)
y = np.array(y_list)

a=[]
i=0
while i<len(y):
    a.append(int(y[i]))
    i=i+1
    
a=np.array(a)


for x in X_list:
    i=0 
    while i<len(x):
        x[i]=float(x[i])
        i=i+1


b=np.array(X_list)
#print('b',b)



import time
start_time = time.time()

model = LogisticRegression2(lr=0.1, num_iter=60000)
model.fit(b, a)
preds = model.predict(b)
print('Precission :',(preds == a).mean())
print("--- %s seconds ---" % (time.time() - start_time))


from sklearn.linear_model import LogisticRegression
start_time = time.time()
log2 = LogisticRegression()
log2.fit(b,a)
print('Precission with sklearn :',log2.score(b,a))
print("--- %s seconds with sklearn ---" % (time.time() - start_time))

