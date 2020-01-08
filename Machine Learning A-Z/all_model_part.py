# import 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values



# encode Categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# avoid dummy variable trap
X=X[:,1:]

#---------------testing and training---------------
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


#---------------feature scaling------- 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


#plot
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('title')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#smooth plot 

x_grid = np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('title')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()



#for single value precdiction
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))



# ----------------------------simple Linear regression ------------------------------
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X,y)

# ------------- multiple linear regression--------
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


# ------------------------ polynomial model ------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression	
poly_reg=PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(x) 
poly_reg.fit(x_poly,y)  
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


# ----------------- SVM -------------------
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)


#---------------decision tree --------------
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(x,y)

#---------------Random forest-------------
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300 ,random_state=0)
regressor.fit(x,y)
