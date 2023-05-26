#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# ## Importing the libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import self_utils as su
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the dataset

# In[3]:


df = pd.read_csv('E:/S4/MML/creditcard.csv')
df.head()


# ## Checking the discrepencies in the data and performing exploratory data analysis

# In[4]:


df.isna().sum()


# In[5]:


df.info()


# In[6]:


len(df)


# In[7]:


df.describe()


# ### Checking the distribution of data

# In[8]:


plt.figure(dpi=100, figsize=(10,6))
sns.countplot(data=df, x='Class')


# ### Checking the effect of `Amount` and `Time` columns of the dataset on `Class`

# In[9]:


plt.figure(dpi=100, figsize=(10,6))
sns.scatterplot(data=df, x='Amount', y='Class', hue='Class')


# In[10]:


plt.figure(dpi=100, figsize=(10,6))
sns.scatterplot(data=df, x='Time', y='Class', hue='Class', )


# ### Dropping the non impactful columns

# In[11]:


df = df.drop(['Time'], axis=1)
df.head()


# ### Scaling `Amount` feature for better results

# In[12]:


sc = StandardScaler()
amount = df['Amount'].values

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
df.head()


# ### Cleaning the dataset by removing any `NaN` or `infinite` values

# In[13]:


df = su.clean_dataset(df)


# ### Checking the Distribution of various features of our dataset

# In[14]:


columns = list(df.columns.values)
columns.remove("Class")
n = 1
t0 = df.loc[df['Class'] == 0]
t1 = df.loc[df['Class'] == 1]

plt.figure()
fig, ax = plt.subplots(12,7,figsize=(16,28))

for i in columns:
    plt.subplot(6,5,n)
    sns.kdeplot(t0[i],label="0")
    sns.kdeplot(t1[i],label="1")
    plt.xlabel(i, fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
    n =n + 1
plt.show()


# ## Preparing the dataset for training

# In[15]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# ### Splitting the dataset into training and validation sets

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.4)
len(X_train)


# ### Checking skewness and using a transformer to mitigate it

# In[17]:


col = X_train.columns
plt.figure(figsize=(20,15))
n=1
for i in col:
    plt.subplot(5,6, n)
    sns.histplot(data = X_train[i])
    n += 1
plt.show()
plt.rcParams.update({'figure.max_open_warning': 0})


# In[18]:


pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
X_train[col] = pt.fit_transform(X_train)
X_test[col] = pt.transform(X_test)


# In[19]:


col = X_train.columns
plt.figure(figsize=(20,15))
n=1
for i in col:
    plt.subplot(5,6, n)
    sns.histplot(data = X_train[i])
    n += 1
plt.show()
plt.rcParams.update({'figure.max_open_warning': 0})


# ### Creating synthetic data using `SMOTE` (Since the dataset is imbalanced)

# In[20]:


smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train.astype('float'), y_train)
print("Before performing smote : ", Counter(y_train))
print("After performing smote : ", Counter(y_train_smote))


#  

#  

#  

# ## Testing various models on the dataset

# ### 1.1. Logistic Regression without synthetic data

# In[20]:


model_ws_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
model_ws_1.fit(X_train, y_train)
y_pred_ws_1 = model_ws_1.predict(X_test)
acc_ws_1 = accuracy_score(y_test, y_pred_ws_1)
acc_ws_1


# #### Confusion Matrix

# In[21]:


su.make_confusion_matrix(y_test, y_pred_ws_1)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[22]:


res11 = su.calculate_results(y_test, y_pred_ws_1)
res11


# ### 1.2 Logistic Regression with synthetic data

# In[23]:


model_s_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
model_s_1.fit(X_train_smote, y_train_smote)
y_pred_s_1 = model_s_1.predict(X_test)
acc_s_1 = accuracy_score(y_test, y_pred_s_1)
acc_s_1


# #### Confusion Matrix

# In[24]:


su.make_confusion_matrix(y_test, y_pred_s_1)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[25]:


res12 = su.calculate_results(y_test, y_pred_s_1)
res12


#  

#  

#  

# ### 2.1. Decision Tree Classifier without synthetic data

# In[26]:


model_ws_2 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_ws_2.fit(X_train, y_train)
y_pred_ws_2 = model_ws_2.predict(X_test)
acc_ws_2 = accuracy_score(y_test, y_pred_ws_2)
acc_ws_2


# #### Confusion Matrix

# In[27]:


su.make_confusion_matrix(y_test, y_pred_ws_2)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[28]:


res21 = su.calculate_results(y_test, y_pred_ws_2)
res21


# ### 2.2. Decision Tree Classifier with synthetic data

# In[29]:


model_s_2 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model_s_2.fit(X_train_smote, y_train_smote)
y_pred_s_2 = model_s_2.predict(X_test)
acc_s_2 = accuracy_score(y_test, y_pred_s_2)
acc_s_2


# #### Confusion Matrix

# In[30]:


su.make_confusion_matrix(y_test, y_pred_s_2)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[31]:


res22 = su.calculate_results(y_test, y_pred_s_2)
res22


#  

#  

#   

# ### 3.1. Naive Bayes Classifier without synthetic data

# In[32]:


model_ws_3 = GaussianNB()
model_ws_3.fit(X_train, y_train)
y_pred_ws_3 = model_ws_3.predict(X_test)
acc_ws_3 = accuracy_score(y_test, y_pred_ws_3)
acc_ws_3


# #### Confusion Matrix

# In[33]:


su.make_confusion_matrix(y_test, y_pred_ws_3)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[34]:


res31 = su.calculate_results(y_test, y_pred_ws_3)
res31


# ### 3.2. Naive Bayes Classifier with synthetic data

# In[35]:


model_s_3 = GaussianNB()
model_s_3.fit(X_train_smote, y_train_smote)
y_pred_s_3 = model_s_3.predict(X_test)
acc_s_3 = accuracy_score(y_test, y_pred_s_3)
acc_s_3


# #### Confusion Matrix

# In[36]:


su.make_confusion_matrix(y_test, y_pred_s_3)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[37]:


res32 = su.calculate_results(y_test, y_pred_s_3)
res32


#  

#  

#  

# ### 4.1. K Nearest Neighbors Classifier without synthetic data

# In[38]:


model_ws_4 = KNeighborsClassifier(n_neighbors=3)
model_ws_4.fit(X_train, y_train)
y_pred_ws_4 = model_ws_4.predict(X_test)
acc_ws_4 = accuracy_score(y_test, y_pred_ws_4)
acc_ws_4


# #### Confusion Matrix

# In[39]:


su.make_confusion_matrix(y_test, y_pred_ws_4)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[40]:


res41 = su.calculate_results(y_test, y_pred_ws_4)
res41


# ### 4.2. K Nearest Neighbors Classifier with synthetic data

# In[41]:


model_s_4 = KNeighborsClassifier(n_neighbors=3)
model_s_4.fit(X_train_smote, y_train_smote)
y_pred_s_4 = model_s_4.predict(X_test)
acc_s_4 = accuracy_score(y_test, y_pred_s_4)
acc_s_4


# #### Confusion Matrix

# In[42]:


su.make_confusion_matrix(y_test, y_pred_s_4)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[43]:


res42 = su.calculate_results(y_test, y_pred_s_4)
res42


#  

#  

#  

# ### 5.1. Random Forest Classifier without synthetic data

# In[44]:


model_ws_5 = RandomForestClassifier(max_depth=5, criterion='entropy')
model_ws_5.fit(X_train, y_train)
y_pred_ws_5 = model_ws_5.predict(X_test)
acc_ws_5 = accuracy_score(y_test, y_pred_ws_5)
acc_ws_5


# #### Confusion Matrix

# In[45]:


su.make_confusion_matrix(y_test, y_pred_ws_5)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[46]:


res51 = su.calculate_results(y_test, y_pred_ws_5)
res51


# #### `Feature importances`

# In[68]:


# Calculate feature importances
importances = model_ws_5.feature_importances_

# Calculate mean decrease in impurity
mean_decrease_impurity = model_ws_5.feature_importances_

# Calculate feature importances
importances = model_s_5.feature_importances_

# Calculate mean decrease in impurity
mean_decrease_impurity = model_s_5.feature_importances_

# Create a bar chart of feature importances
plt.bar(X_train.columns, importances)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()


# #### `SHAP`

# In[72]:


import shap

# Create a SHAP explainer object
explainer = shap.TreeExplainer(model_ws_5)

# Calculate SHAP values for a single instance
instance = X_test.iloc[0]
shap_values = explainer.shap_values(instance)

# Visualize the SHAP values
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], instance)


# ### 5.2. Random Forest Classifier with synthetic data

# In[47]:


model_s_5 = RandomForestClassifier(max_depth=5, criterion='entropy')
model_s_5.fit(X_train_smote, y_train_smote)
y_pred_s_5 = model_s_5.predict(X_test)
acc_s_5 = accuracy_score(y_test, y_pred_s_5)
acc_s_5


# #### Confusion Matrix

# In[48]:


su.make_confusion_matrix(y_test, y_pred_s_5)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[49]:


res52 = su.calculate_results(y_test, y_pred_s_5)
res52


# #### `Feature importances`

# In[67]:


# Calculate feature importances
importances = model_s_5.feature_importances_

# Calculate mean decrease in impurity
mean_decrease_impurity = model_s_5.feature_importances_

# Create a bar chart of feature importances
plt.bar(X_train.columns, importances)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()


# #### `SHAP`

# In[70]:


import shap

# Create a SHAP explainer object
explainer = shap.TreeExplainer(model_s_5)

# Calculate SHAP values for a single instance
instance = X_test.iloc[0]
shap_values = explainer.shap_values(instance)

# Visualize the SHAP values
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], instance)


#  

#  

# ### 6.1. Support Vector Classifier without synthetic data

# In[50]:


model_ws_6 = SVC()
model_ws_6.fit(X_train, y_train)
y_pred_ws_6 = model_ws_6.predict(X_test)
acc_ws_6 = accuracy_score(y_test, y_pred_ws_6)
acc_ws_6


# #### Confusion Matrix
# 

# In[51]:


su.make_confusion_matrix(y_test, y_pred_ws_6)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[52]:


res61 = su.calculate_results(y_test, y_pred_ws_6)
res61


# ### 6.2. Support Vector Classifier with synthetic data

# In[53]:


model_s_6 = SVC()
model_s_6.fit(X_train_smote, y_train_smote)
y_pred_s_6 = model_s_6.predict(X_test)
acc_s_6 = accuracy_score(y_test, y_pred_s_6)
acc_s_6


# #### Confusion Matrix

# In[54]:


su.make_confusion_matrix(y_test, y_pred_s_6)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[55]:


res62 = su.calculate_results(y_test, y_pred_s_6)
res62


# #### `Permutation importance`

# In[ ]:


from sklearn.inspection import permutation_importance

# Calculate feature importances using the 'permutation_importance' function
importances = permutation_importance(model_s_6, X_test, y_test, n_repeats=10, random_state=0)

# Sort features by importance
sorted_importances = sorted(enumerate(importances.importances_mean), key=lambda x: x[1], reverse=True)

# Separate feature names and importance values
features, importance_values = zip(*sorted_importances)

# Create a bar chart of feature importances
plt.bar(features, importance_values)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()


#  

#  

# ### 7.1. XGBoost Classifier without synthetic data

# In[21]:


model_ws_7 = XGBClassifier()
model_ws_7.fit(X_train, y_train)
y_pred_ws_7 = model_ws_7.predict(X_test)
acc_ws_7 = accuracy_score(y_test, y_pred_ws_7)
acc_ws_7


# #### Confusion Matrix

# In[22]:


su.make_confusion_matrix(y_test, y_pred_ws_7)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[23]:


res71 = su.calculate_results(y_test, y_pred_ws_7)
res71


# ### 7.2. XGBoost Classifier with synthetic data

# In[24]:


model_s_7 = XGBClassifier()
model_s_7.fit(X_train_smote, y_train_smote)
y_pred_s_7 = model_s_7.predict(X_test)
acc_s_7 = accuracy_score(y_test, y_pred_s_7)
acc_s_7


# #### Confusion Matrix

# In[25]:


su.make_confusion_matrix(y_test, y_pred_s_7)


# #### `Accuracy`, `f1Score`, `precision` and `recall` of the model

# In[26]:


res72 = su.calculate_results(y_test, y_pred_s_7)
res72


# #### `Feature Importances`

# In[75]:


# Calculate feature importances
importances = model_s_7.feature_importances_

# Create a bar chart of feature importances
plt.bar(X_train.columns, importances)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()


# #### `Feature Importances (Weight)`

# In[77]:


# Calculate feature importances using the 'weight' method
importances = model_s_7.get_booster().get_score(importance_type='weight')

# Sort features by importance
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

# Separate feature names and importance values
features, importance_values = zip(*sorted_importances)

# Create a bar chart of feature importances
plt.bar(features, importance_values)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance (Weight)")
plt.title("Feature Importances (Weight)")
plt.show()


# #### `Feature Importances (Cover)`

# In[78]:


# Calculate feature importances using the 'cover' method
importances = model_s_7.get_booster().get_score(importance_type='cover')

# Sort features by importance
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

# Separate feature names and importance values
features, importance_values = zip(*sorted_importances)

# Create a bar chart of feature importances
plt.bar(features, importance_values)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance (Cover)")
plt.title("Feature Importances (Cover)")
plt.show()


# #### `Feature Importances (gain)`

# In[79]:


# Calculate feature importances using the 'gain' method
importances = model_s_7.get_booster().get_score(importance_type='gain')

# Sort features by importance
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

# Separate feature names and importance values
features, importance_values = zip(*sorted_importances)

# Create a bar chart of feature importances
plt.bar(features, importance_values)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance (Gain)")
plt.title("Feature Importances (Gain)")
plt.show()


# # Comparing precision, accuracy f1 score and recall of all the models

# In[62]:


dp = pd.DataFrame([res11,res12,res21,res22,res31,res32,res41,res42,res51,res52,res61,res62,res71,res72],index=['1.1','1.2','2.1','2.2','3.1','3.2','4.1','4.2','5.1','5.2','6.1','6.2','7.1','7.2'])


# In[63]:


dp


#  

#   

#  

# ## Save the Perfect Model

# In[27]:


import pickle

filename = 'fraud_detection.sav'
pickle.dump(model_s_7, open(filename, 'wb'))


# In[ ]:




