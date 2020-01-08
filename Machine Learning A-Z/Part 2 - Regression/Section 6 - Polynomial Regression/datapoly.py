import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# ------------------------ polynomial model ------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression	
poly_reg=PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(x) 
poly_reg.fit(x_poly,y)  
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('poly reg')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


#predict 
#here we get error of array so we use [[]]

