import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import random

data = pd.read_csv(r"C:\Users\DELL\Desktop\CSE assignment\Gym.csv")
print(data.describe())
print(data.info())
dummies = pd.get_dummies(data,columns=['Workout_Type'],dtype='int64')
print(dummies)
data1 = dummies.drop(columns=['Age'])
print(data1)
print(data1.columns)
#X = data1 [['Experience_Level', 'BMI','Water_Intake (liters)','Fat_Percentage','Avg_BPM']]
#X = data1 [['Water_Intake (liters)','Fat_Percentage','Avg_BPM']]
#X = data1 [['BMI','Water_Intake (liters)','Fat_Percentage']]
#X = data1 [['Experience_Level', 'BMI','Water_Intake (liters)']]
X = data1 [['Fat_Percentage','Avg_BPM']]
Y = data1 ['Calories_Burned']
print(X)
print(Y)
sns.scatterplot(data1)
plt.plot(X,Y)
plt.show()
random.seed(1)
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size= .30)


regr = LinearRegression()
regr.fit(X_train, Y_train)
print(regr.score(X_test, Y_test))





