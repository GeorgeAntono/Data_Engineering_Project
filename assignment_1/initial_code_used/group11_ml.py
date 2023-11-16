import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pickle

dataset = pd.read_csv('churn.csv')
dataset.head(10)


dataset.isna().sum()

label_encoder = preprocessing.LabelEncoder()
for i in ['Geography', 'Gender']:
    dataset[i]= label_encoder.fit_transform(dataset[i])
dataset.loc[:5 , ['Geography', 'Gender']]



X_Columns = ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']

Y_columns= ['Exited']

X = dataset.copy()[X_Columns]
y = dataset.copy()[Y_columns]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


model_rf_class =RandomForestClassifier()


model_RFC= RandomForestClassifier(n_estimators=300,max_depth=10,criterion='gini')
model_RFC.fit(X_train,y_train.values.ravel())

file_name = 'model.pkl'

pickle.dump(model_RFC, open(file_name, 'wb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))
predictions_rf_class_pickled = loaded_model.predict(X_test)
print(balanced_accuracy_score(y_test, predictions_rf_class_pickled))


scaler = StandardScaler()



scaler.fit(X_train)


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled,columns = X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns = X.columns) 

model_SVC = SVC(class_weight='balanced')

model_SVC_best= SVC(C=1,kernel='rbf', gamma='auto',class_weight='balanced')
model_SVC_best.fit(X_train_scaled, y_train.values.ravel())

file_name = 'modelSVC.pkl'
pickle.dump(model_SVC_best, open(file_name, 'wb'))


loaded_model2 = pickle.load(open('modelSVC.pkl', 'rb'))
y_predictions_SVC_pickled = loaded_model2.predict(X_test_scaled)
print(balanced_accuracy_score(y_test, y_predictions_SVC_pickled))





