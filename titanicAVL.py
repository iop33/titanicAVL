import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')
X_columns = data.columns.tolist()
data2 = pd.read_csv('test.csv')

features = ["Pclass", "Sex", "SibSp", "Parch","Age"]


avg_values = data["Age"].mean()  # Izračunavanje prosečnih vrednosti za svaku kolonu
data["Age"] =data["Age"].fillna(avg_values) 

avg_values = data2["Age"].mean()  # Izračunavanje prosečnih vrednosti za svaku kolonu
data2["Age"] =data2["Age"].fillna(avg_values) 

print(data["Age"])

y_train = data['Survived']
x_train = pd.get_dummies(data[features])
y_test = data2['Survived']
x_test = pd.get_dummies(data2[features])
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=1)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
rez=metrics.accuracy_score(y_test, y_predict)
print("% tacnost je",rez)
plt.figure(figsize=(60, 30))
tree.plot_tree(model, fontsize=12, feature_names=list(X_columns), filled=True, rounded=True, class_names=['0','1'])

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
rez=metrics.accuracy_score(y_test, y_predict)
print("% tacnost sume je",rez)