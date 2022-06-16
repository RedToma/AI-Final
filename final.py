from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV      
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

dd = pd.read_csv('C:/Users/jimin/Desktop/data/diabetes.csv')

d_input = dd[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
              'DiabetesPedigreeFunction','Age']].to_numpy()
d_target = dd['Outcome'].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(d_input,d_target,train_size=0.7)

param_grid = {'C': [1,5,10,20,30,40,50,60,70,80,90,100],
             'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)

grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

ss = StandardScaler()
ss.fit(x_train)

x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

s = SVC(gamma=0.001, C=10, kernel='rbf')

s.fit(x_train,y_train)
res = s.predict(x_test)

conf=np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

no_correct=0
for i in range(2):
    no_correct+=conf[i][i]
accuracy=no_correct/len(res)
print("훈련 세트 정확률: {}".format(s.score(x_train,y_train)*100))
print("테스트 세트 정확률: {}" .format(accuracy*100))

