import pandas as pd
import math
import numpy as np
import time

df = pd.read_csv('test.csv')
#print(df.head())


from io import StringIO

col = ['publication', 'content']
df = df[col]
df.columns = ['publication', 'content']


df['category_id'] = df['publication'].factorize()[0]
#print(df.head())

category_id_df = df[['publication', 'category_id']].drop_duplicates().sort_values('category_id')

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'publication']].values)
#print(df.head())

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('publication').content.count().plot.bar(ylim=0)
#plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.content).toarray()
labels = df.category_id
#print(features.shape)


from sklearn.feature_selection import chi2
import numpy as np

N = 2
for publisher, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#  print("# '{}':".format(publisher))
#  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

X_train = df['content'] 
X_test = df['content']
y_train = df['publication']
y_test = df['publication']
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

start_time = time.time()
log = LogisticRegression().fit(X_train_tfidf, y_train)
print('Score with sklearn = ',log.score(X_train_tfidf, y_train))
print("--- %s seconds with sklearn ---" % (time.time() - start_time))

from scipy.sparse import hstack
from scipy import sparse

class LogisticRegression2:
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, theta=0, verbose=False):
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
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            #print('type X.T',type(X.T[0][0]))
            #print('type y',type(y[0]))
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


X_train_dense = X_train_tfidf.toarray()
y_train_dence = y_train.as_matrix()


y=np.zeros(382)
for elem in y_train_dence:
    # New York Times  =  1
    # Breitbart  =  2
    # CNN  =  3
    # Business Insider  =  4
    
    if elem == 'New York Times':
        np.append(y,1.0)
    elif elem == 'Breitbart':
        np.append(y,2.0)
    elif elem == 'CNN':
        np.append(y,3.0)
    elif elem == 'Business Insider':
        np.append(y,4.0)

    
#print('1',X_train_tfidf)
#print('2',X_train_tfidf.toarray())
start_time = time.time()
model = LogisticRegression2(lr=0.1, num_iter=100)
model.fit(X_train_dense, y)
preds = model.predict(X_train_dense)
print('Score without sklearn :',(preds == y).mean())
print("--- %s seconds ---" % (time.time() - start_time))


