import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values
y=y.reshape(10,1)

#---------------feature scaling------- 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# ----------------- SVM -------------------
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('title')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#feature scaling need to inverse
# y is giving error is array shape hence
# kernal type = linear (for linear) , rbf(for non linear) , poly(for polynomial)
# here
# sc_x.transform --- is using feature scaling on 6.5 which is not done earlier
# inverse_transform --because we apply scaling on scaled Y , not actual 