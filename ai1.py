import matplotlib.pyplot as plt, numpy as np
from pandas import read_csv
from sklearn import linear_model
from itertools import combinations
from statistics import mode

data = None
with open('BLK.csv') as csv_file:
    data = read_csv(csv_file)


title = [ 
    'volume',
    'open',
    'high', 
    'low', 
    'adjclose']


x = data[title].as_matrix()
y = data['close'].as_matrix()
    
n = len(title)


for i in range(0, n):
    plt.xlabel(title[i])
    plt.ylabel('close')
    plt.plot(data[title[i]].as_matrix(), y, 'ro')

    plt.show
    plt.savefig('close-{}.svg'.format(title[i]).replace(' ', '-'))
    plt.clf()



##Характеристики цены закрытия

#Cреднее значение цены закрытия
sum = sum(data['close'])
print(sum)
num = len(data['close'])
print(num)
avg = sum/num
print('Средняя цена',avg)


#Медиана
middle = len(data['close'])/2+0.5
list_sorted=sorted(data['close'])
mediane = list_sorted[int(middle)]
print('Медиана',mediane)


#Максимум
max = max(data['close'])
print('Максимум',max)

#Минимум
min = min(data['close'])
print('Минимум',min)

#Мода
mode = mode(data['close'])
print('Мода',mode)

#Размах
r = max - min
print('Размах', r)

#Стандартное отклонение
so = np.std(data['close'])
print('Стандартное отклонение',so)

#Дисперсия
d = so * so
print('Дисперсия',d)

