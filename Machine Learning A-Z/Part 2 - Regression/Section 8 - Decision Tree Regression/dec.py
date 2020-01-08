import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values
print(dataset)
print(x)
print(y)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(x,y)

print(regressor.predict([[6.5]]))

x_grid = np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('title')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()




