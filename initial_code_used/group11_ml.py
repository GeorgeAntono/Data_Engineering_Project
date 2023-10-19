#!/usr/bin/env python
# coding: utf-8

# 0) Libraries

# In[2]:


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

# 1) Download and read the data from this link:  
# https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
# 
# *Main function: pd.read_csv*
# 

# In[3]:


dataset = pd.read_csv('churn.csv')
dataset.head(10)


# In[4]:


dataset.isna().sum()


# 2) Convert categorical variables (Geography,Gender) to Numerical 
# 
# *Main function: label encoding*

# In[5]:


label_encoder = preprocessing.LabelEncoder()


# In[6]:


for i in ['Geography', 'Gender']:
    dataset[i]= label_encoder.fit_transform(dataset[i])
dataset.loc[:5 , ['Geography', 'Gender']]


# 3) Split X και y, train και test
# 
# X_Columns: 
# CreditScore, Geography, Gender, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary 
# 
# Y_columns: 
# Exited
# 
# and add stratify = y στα parameters από το train_test_split
# 
# *Main function: (train_test_split)*

# In[7]:


X_Columns = ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']


# In[8]:


Y_columns= ['Exited']


# In[9]:


X = dataset.copy()[X_Columns]
y = dataset.copy()[Y_columns]
print(X.shape)
print(y.shape)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# 4) Train with these characteristics:
# 
# * Random Forest 
# * Without PCA
# * Without clusters
# * With Precision scoring, 
# * Try with grid search 200, 300 και 500 tress, with max depth 3,5 and 10, as well as the two avaible criterion. 
# 
# *Main function: grid_search, RandomForestClassifier*

# In[11]:


model_rf_class =RandomForestClassifier()


# In[29]:


"""param_grid_class = {'n_estimators': [200,300,500],
              'max_depth' : [3,5,10],
              'criterion' :['gini', 'entropy']
             } """


# In[30]:


"""grid_search_rf_class = GridSearchCV(estimator=model_rf_class, 
                              param_grid=param_grid_class, 
                              scoring= 'accuracy',
                              cv=5,
                             verbose=1)"""


# In[49]:


#grid_search_rf_class.fit(X_train,y_train.values.ravel())


# In[ ]:


#grid_search_rf_class.best_params_


# 5) Export from the model random forest a data frame with the feature importance with the best model on each cluster. 
# 
# *Main function: feature_importance_*

# In[ ]:


"""pd.DataFrame(data = grid_search_rf_class.best_estimator_.feature_importances_,
            index = X.columns,
            columns=['feature_importance']).sort_values(by='feature_importance',
                                                       ascending =False)"""


# In[12]:


model_RFC= RandomForestClassifier(n_estimators=300,max_depth=10,criterion='gini')
model_RFC.fit(X_train,y_train.values.ravel())

file_name = 'model.pkl'

pickle.dump(model_RFC, open(file_name, 'wb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))
predictions_rf_class_pickled = loaded_model.predict(X_test)
balanced_accuracy_score(y_test, predictions_rf_class_pickled)


# 6) Train with these characteristics:
# * Algorithm Support Vector Machine
# * With Standard Scaler
# * With scoring Balanced Accuracy
# * Try only with the kernel Radial basis function, to find gamma automatically,  while  the parameter C takes the values 0.001, 0.01, 1, 10 ,100 ,1000
# 
# *Main function: grid_search, SVC(class_weight= 'balanced'), StandarScaler, balanced_accuracy*
# 

# In[20]:


scaler = StandardScaler()


# In[36]:


scaler.fit(X_train)


# In[37]:


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[38]:


X_train_scaled = pd.DataFrame(X_train_scaled,columns = X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns = X.columns) 


# In[39]:


model_SVC = SVC(class_weight='balanced')


# In[40]:


"""tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': ["auto"],
                     'C': [0.001, 0.01, 1, 10, 100, 1000]}
                   ]"""


# In[41]:


"""model_SVC_grid = GridSearchCV(estimator  =  model_SVC,
                              param_grid = tuned_parameters,
                              scoring="balanced_accuracy",
                              cv=5,
                              verbose = True
                             )"""


# In[42]:


#model_SVC_grid = model_SVC_grid.fit(X_train_scaled, y_train.values.ravel())  #DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel(). y = column_or_1d(y, warn=True)


# In[43]:


#model_SVC_grid.best_params_


# In[44]:


#pd.DataFrame(model_SVC_grid.cv_results_).sort_values('rank_test_score')


# In[45]:


model_SVC_best= SVC(C=1,kernel='rbf', gamma='auto',class_weight='balanced')
model_SVC_best.fit(X_train_scaled, y_train.values.ravel())


# In[46]:


import pickle
file_name = 'modelSVC.pkl'
pickle.dump(model_SVC_best, open(file_name, 'wb'))


# In[47]:


#loaded_model2 = pickle.load(open('modelSVC.pkl', 'rb'))
#y_predictions_SVC_pickled = loaded_model2.predict(X_test_scaled)
#balanced_accuracy_score(y_test, y_predictions_SVC_pickled)


# 7) Predict the two models from the test sample, then export Confusion Matrix, Accuracy and Balanced_Accuracy.
# 
# *Main function: predict, confusion_matrix, accuracy, balanced_accuracy*
# 

# **Random Forest**

# In[ ]:


#predictions_rf_class = grid_search_rf_class.predict(X_test)


# In[ ]:


#confusion_matrix(y_test,predictions_rf_class)


# In[ ]:


#accuracy_score(y_test,predictions_rf_class)


# In[ ]:


#balanced_accuracy_score(y_test,predictions_rf_class)


# **Support Vector Machine**

# In[ ]:


#y_predictions_SVC = model_SVC_grid.predict(X_test_scaled)


# In[ ]:


#confusion_matrix(y_test, y_predictions_SVC)


# In[ ]:


#accuracy_score(y_test, y_predictions_SVC)


# In[ ]:


#balanced_accuracy_score(y_test, y_predictions_SVC)


# In[ ]:




