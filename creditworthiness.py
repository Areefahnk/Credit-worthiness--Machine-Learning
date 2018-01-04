
# coding: utf-8

# In[729]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

creditData = pd.read_csv('creditData.csv')


# # DATA PRE-PROCESS

# In[730]:


#print(creditData.shape)


# In[731]:


#print(creditData.columns)


# In[732]:


creditData.isnull().sum()


# In[733]:


#creditData.info()


# In[734]:


print("Data types and their frequency\n{}".format(creditData.dtypes.value_counts()))


# In[735]:


object_columns_creditData = creditData.select_dtypes(include=['object'])
print(object_columns_creditData.iloc[0])


# In[736]:


cols = ['checking_status', 'credit_history','purpose', 'savings_status', 'employment', 'personal_status',
        'other_parties', 'property_magnitude','other_payment_plans', 'housing', 'job', 'own_telephone',
       'foreign_worker', 'class']
for name in cols:
    print(name,':')
    print(object_columns_creditData[name].value_counts(),'\n')


# In[737]:


#ordinal values
checking_status_dict = {
    'checking_status': {
        ">=200": 3,
        "0<=X<200": 2,
        "<0": 1,
        """'no checking'""": 0
 
    }}
savings_status_dict = { 
    'savings_status':{
         """'no known savings'""": 0,
        "<100": 1,
        "100<=X<500": 2,
        "500<=X<1000": 3,
        ">=1000": 4
    }}
employment_dict = {
      'employment': {
        """unemployed""": 0,
        "<1": 1,
        "1<=X<4": 2,
        "4<=X<7": 3,
        ">=7": 4
 
    }}

job_dict = {
'job' :{
    """'unemp/unskilled non res'""":0,
    """'unskilled resident'""":1,
    """skilled""":2,
    """'high qualif/self emp/mgmt'""":3
}}

credit_history_dict={
    'credit_history':
    {
    """'existing paid'""":2,
    """'critical/other existing credit'""":4,
    """'delayed previously'""":3,
    """'all paid'""":1,  
    """'no credits/all paid'""":0
}}
creditData = creditData.replace(checking_status_dict)
creditData = creditData.replace(savings_status_dict)
creditData = creditData.replace(employment_dict)
creditData = creditData.replace(job_dict)
creditData = creditData.replace(credit_history_dict)
creditData[['checking_status','savings_status','employment','job','credit_history']].head()


# In[738]:


nominal = ["purpose","personal_status","other_parties","property_magnitude","other_payment_plans","housing","own_telephone","foreign_worker"]

for col in nominal:
    creditData[col] = creditData[col].astype('category')
#creditData.info()


# In[137]:


#creditData[nominal] = creditData[nominal].apply(lambda x: x.cat.codes)


# In[739]:


#nominal values
#nominal = ["credit_history","purpose","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker","class"]
creditData_dummy = creditData.drop(['age','duration','credit_amount','class'], axis=1)
dummy_creditData = pd.get_dummies(creditData_dummy[nominal])
creditData_dummy= pd.concat([creditData_dummy, dummy_creditData], axis=1)
creditData_dummy = creditData_dummy.drop(nominal, axis=1)


# In[740]:


#le = preprocessing.LabelEncoder()
#nominal = ["credit_history","purpose","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker","class"]
#le.fit(creditData[nominal])


# In[741]:


creditData.info()


# In[742]:


#creditData.to_csv("processed_creditData.csv",index=False)


# # SPLIT

# In[743]:


creditData_X = creditData_dummy
creditData_y = creditData[['class']]
creditData_y
creditData_y=creditData_y.replace({"class":{'good':1,'bad':0}})
X_train, X_test, y_train, y_test = train_test_split(creditData_X, creditData_y, test_size = .4, random_state=43)


# # Feature selection

# In[744]:


# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 15)
fit = rfe.fit(creditData_X , creditData_y)
print("number of features")
print(fit.n_features_) 
#print("selected features")
#print(fit.support_) 
#print("features ranking")
#print(creditData_X.columns)
#print(fit.ranking_)


# In[745]:


fit.get_params()
dictionary = dict(zip(creditData_X.columns, fit.ranking_))
dictionary


# In[746]:


#creditData_X = creditData_X.iloc[:, creditData_X.columns not in ['age','duration','residence_since','credit_amount']]
#creditData_X = creditData_X.drop(['age','duration','credit_amount'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(creditData_X, creditData_y, test_size = .4, random_state=43)
#y_train
#y_train.shape


# # LOGISTIC REGRESSION MODEL

# In[747]:


LogReg = LogisticRegression()
lr = LogReg.fit(X_train, y_train)


# In[748]:


lr_y_pred = LogReg.predict(X_test)


# In[749]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, lr_y_pred)
confusion_matrix
#report = classification_report(y_test, lr_y_pred)


# In[750]:


print("LOGISTIC REGRESSION Report:")
print(classification_report(y_test, lr_y_pred))
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,lr_y_pred)
y_prob = LogReg.predict_proba(X_test)[:, 1]
print("AUC score: {:.2f}".format(metrics.roc_auc_score(y_test, y_prob)))


# # LinearSVC

# In[753]:


from sklearn.svm import LinearSVC
s_clf = LinearSVC(random_state=0)
svc = s_clf.fit(X_train, y_train)
svc_y_pred = s_clf.predict(X_test)


# In[754]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, svc_y_pred)
confusion_matrix


# In[755]:


print("LinearSVC Report:")
print(classification_report(y_test, svc_y_pred))
print("Training set score: {:.2f}".format(s_clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(s_clf.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,lr_y_pred)
y_svc = s_clf.decision_function(creditData_X).ravel()
print ("AUC score: {:.2f}".format(metrics.roc_auc_score(creditData_y, y_svc)))


# # Decision Tree

# In[756]:


from sklearn import tree
d_clf = tree.DecisionTreeClassifier()
dt_clf = d_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)


# In[757]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, dt_y_pred)
confusion_matrix


# In[758]:


print("Decision Tree Report:")
print(classification_report(y_test, dt_y_pred))
print("Training set score: {:.2f}".format(dt_clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(dt_clf.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,dt_y_pred)
y_dt_prob = dt_clf.predict_proba(X_test)[:, 1]
print("AUC score: {:.2f}".format(metrics.roc_auc_score(y_test, y_dt_prob)))


# # Random Forest

# In[759]:


from sklearn.ensemble import RandomForestClassifier
regr = RandomForestClassifier(max_depth=2, random_state=0)
rf = regr.fit(X_train, y_train)


# In[760]:


#print(regr.feature_importances_)


# In[761]:


rf_y_pred = regr.predict(X_test)


# In[762]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, rf_y_pred)
confusion_matrix


# In[763]:


importances = regr.feature_importances_
std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Features:")

for f in range(5):
    print("%d. feature %s (%f)" % (f + 1, str(X_train.columns[f]), importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(5), importances[0:5],
        color="r", yerr=std[0:5], align="center")
plt.xticks(range(5), X_train.columns[0:5])
plt.show()


# In[764]:


print("Random Forest Report:")
print(classification_report(y_test, rf_y_pred))
print("Training set score: {:.2f}".format(rf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(rf.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,rf_y_pred)
y_rf_prob = rf.predict_proba(X_test)[:, 1]
print("AUC score: {:.2f}".format(metrics.roc_auc_score(y_test, y_rf_prob)))


# #  Multi-Layered Perceptron (Neural Network)

# In[765]:


from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
mlp_clf.fit(X_train, y_train)


# In[766]:


mlp_y_pred = mlp_clf.predict(X_test)


# In[767]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, mlp_y_pred)
confusion_matrix


# In[769]:


print("Multi-Layered Perceptron (Neural Network) Report:")
print(classification_report(y_test, mlp_y_pred))
print("Training set score: {:.2f}".format(mlp_clf.score(X_train, y_train)))
print("Test set score: {:.2f}".format(mlp_clf.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,mlp_y_pred)
y_mlp_prob = mlp_clf.predict_proba(X_test)[:, 1]
print("AUC score: {:.2f}".format(metrics.roc_auc_score(y_test, y_mlp_prob)))

