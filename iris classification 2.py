#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# # Loading The Data

# In[12]:


df = pd.read_csv(r"D:\iris classification\Iris.csv")
df.head()


# In[13]:


# data stats
df.describe()


# In[14]:


df.info()


# In[15]:


df['Species'].value_counts()


# In[16]:


df.shape


# # Data Pre-Processing

# In[17]:


# checking null value
df.isnull().sum()


# # Exploratory Analysis

# In[18]:


sns.histplot(df['SepalLengthCm'])
plt.title('Sepal Length Histogram')
plt.show()


# In[19]:


sns.histplot(df['SepalWidthCm'])
plt.title('Sepal Width Histogram')
plt.show()


# In[20]:


sns.histplot(df['PetalLengthCm'])
plt.title('Petal Length Histogram')
plt.show


# In[21]:


sns.histplot(df['PetalWidthCm'])
plt.title('Petal Width Histogram')
plt.show


# In[22]:


# scatterplot
color=['red','Blue','Orange']
species=['Iris-virginica','Iris-versicolor','Irisetosa']


# In[23]:


for i in range (3):
    x=df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=color[i],label=species[i])
    
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()


# In[24]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = color[i], label = species[i])
    
plt.xlabel("petal Length")
plt.ylabel("petal Width")
plt.legend()


# In[25]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = color[i], label = species[i])
    
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[26]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = color[i], label = species[i])
    
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# # Correlation Matrix

# In[27]:


numeric_df = df.select_dtypes(include=[np.number])
numeric_df.corr()


# In[28]:


corr = numeric_df.corr()
fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(corr, annot = True, ax = ax)


# # Label Encoder

# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])


# In[30]:


# Split data into features and target
X = df.drop(columns=["Species"])
Y = df['Species']


# # Model Training

# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)


# In[34]:


#random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
print("Classification Report:", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred_rf))


# In[35]:


#hyper parameter tuned random forest
from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],     
    'min_samples_split': [2, 5, 10],          
    'min_samples_leaf': [1, 2, 4],             
    'bootstrap': [True, False]                 
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# In[36]:


# logistic regression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) 

print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[37]:


# hyperparameter tuned logistic regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],                    
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],    
    'solver': ['liblinear', 'saga'],                 
    'max_iter': [100, 200, 500],                      
}

log_reg = LogisticRegression()

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_log_reg = grid_search.best_estimator_

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# In[38]:


# knn - K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[39]:


# hyperparameter tuned KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9],            
    'weights': ['uniform', 'distance'],     
    'metric': ['euclidean', 'manhattan', 'minkowski'] 
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_knn = grid_search

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[43]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[44]:


#hyperparameter tuned decision tree
param_grid = {
    'criterion': ['gini', 'entropy'],         
    'max_depth': [5, 10, 20, None],           
    'min_samples_split': [2, 5, 10],         
    'min_samples_leaf': [1, 2, 4],           
    'max_features': [None, 'auto', 'sqrt', 'log2']  
}

dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_dt = grid_search.best_estimator_

y_pred = best_dt.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[45]:


#SVM
from sklearn.svm import SVC

model = SVC(kernel='linear', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[46]:


#hyperparameter tuned SVM
param_grid = {
    'C': [0.1, 1, 10, 100],                  
    'kernel': ['linear', 'rbf', 'poly'],      
    'gamma': ['scale', 'auto'],               
    'degree': [2, 3, 4],               
}
svc = SVC() 

grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)  
grid_search.fit(x_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_svm = grid_search.best_estimator_


y_pred = best_svm.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:




