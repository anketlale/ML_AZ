# importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv')

print("dataset:",dataset)
print()
print()

#------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print("X =",X)
print()
print("Y =",y)
print()

# 1)-------------------handling missing data-------------
from sklearn.impute import SimpleImputer
#sklearn library's imputer use to handle missing values 
#1.create object of class
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
#--axis =0 means column ( mean )
#2.fit in our data 
imputer = imputer.fit(X[:,1:3])
#--index is 1 , 2 to take in fit 
#3.
X[:, 1:3] = imputer.transform(X[:, 1:3])
#4.print x
print("handling missing x=",X)
print()


# 2)-------------------encode Categorical data --------------
#------------------dependant :
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#clas of sklearn for category
#this make value encoded  

#1.make object
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')



X = np.array(ct.fit_transform(X), dtype=np.float)
#----------------------Encoding the Dependent Variable :
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)

print("encoding x=")

print(X)

print()

print("encoding y=")

print(y)


# 3)--------------------------------make test data and train data---------------------------------------------- 

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

print("X_train=",X_train)
print("X_test=",X_test)
print("y_train=",y_train)
print("y_test=",y_test)

# 4)--------------------------------feature scaling----------------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


print("X_train scaling=",X_train)

print("X_test scaling=",X_test)

