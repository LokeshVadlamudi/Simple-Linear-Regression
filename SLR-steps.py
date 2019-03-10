
# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import database
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


#splitting into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size = 1/3,random_state = 0)


#fitting simple linear regression to train set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the Y test
Y_pred = regressor.predict(X_test)
print(Y_pred)
print(Y_test)

#seeing the train set results

# plt.scatter(X_train,Y_train,color = 'red')
# plt.plot(X_train,regressor.predict(X_train),color = 'blue')
# plt.title('Salary Vs Experience')
# plt.xlabel('experience')
# plt.ylabel('salary')
# plt.show()


#seeing the test set results

plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary Vs Experience')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()




