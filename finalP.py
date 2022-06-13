from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

dd = pd.read_csv('C:/Users/jimin/Desktop/data/diabetes.csv')

d_input = dd[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
              'DiabetesPedigreeFunction','Age']].to_numpy()
d_target = dd['Outcome'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(d_input,d_target,train_size=0.7)

p = Perceptron(max_iter=100, eta0=0.001, verbose=0)
p.fit(x_train,y_train)
res = p.predict(x_test)

conf=np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

no_correct=0
for i in range(2):
    no_correct+=conf[i][i]
accuracy=no_correct/len(res)
print("훈련 세트 정확률: {}".format(p.score(x_train,y_train)*100))
print("테스트 세트 정확률: {}" .format(accuracy*100))