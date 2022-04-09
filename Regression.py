import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x = np.array([2,5,3,10])
y = np.array([8,25,9,40])

x_new = np.array([np.ones(len(x)),x.flatten()]).T

theta_best_values=np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y)
 
print(theta_best_values)

predict_value = x_new.dot(theta_best_values)

plt.scatter(x,y,s=30,marker='o')
plt.plot(x_new,predict_value,c='green')
plt.xlabel("Feature_1")
plt.ylabel("Target_Variable")
plt.title('Simple Linear Regression')
plt.show()
