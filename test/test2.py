import math 
import data
import numpy as np


x = np.array([[4.0, 2.0, 0.60],
              [4.2, 2.1, 0.59],
              [3.9, 2.0, 0.58],
              [4.3, 2.1, 0.62],
              [4.1, 2.2, 0.63]])


x = np.array([[4.0, 2.0, 0.60],
              [4.2, 2.1, 0.59],
              [3.9, 2.0, 0.58]])

y = x.transpose()
print(np.cov(x))
print(y)
for list in y:
    print(list.mean())
    print(list.std())


print(np.cov(y))
print(np.linalg.inv(np.cov(y)))
print(np.dot(np.cov(y),np.linalg.inv(np.cov(y))))

print(np.dot(x,np.linalg.inv(x)))
print((x-y))
print((x-y).transpose())
