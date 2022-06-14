from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

dd = pd.read_csv('C:/Users/jimin/Desktop/data/diabetes.csv')

d_input = dd[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
              'DiabetesPedigreeFunction','Age']].to_numpy()
d_target = dd['Outcome'].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(d_input,d_target,train_size=0.6)

param_grid = {'hidden_layer_sizes': [(100),(100,80),(100,80,20)],
             'learning_rate_init': [0.001, 0.01, 0.1, 1, 10, 100],
             'batch_size': [30, 60, 100, 200, 300],
             'max_iter':[100,200,300,400,500]}

mlp = MLPClassifier()

grid_search = GridSearchCV(mlp, param_grid, cv=5, return_train_score=True)

grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

ss = StandardScaler()
ss.fit(x_train)

x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

mlp=MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',
                  learning_rate_init=0.01,batch_size=60,max_iter=500,
                  solver='sgd',verbose=True)

mlp.fit(x_train,y_train)

res = mlp.predict(x_test)

conf=np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

no_correct=0
for i in range(2):
    no_correct+=conf[i][i]
accuracy=no_correct/len(res)
print("훈련 세트 정확률: {}".format(mlp.score(x_train,y_train)*100))
print("테스트 세트 정확률: {}" .format(accuracy*100))