import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# ----------------------------simple Linear regression ------------------------------
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X,y)

#plot
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('title')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()















# ----------- notes -------------
# print("dataset:")
# print(dataset)
# print()
