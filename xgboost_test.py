#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# In[ ]:


data_3_train = pd.read_csv('D:\\79886\\Documents\\HKUST\\Courses\\5001 Foundations of Data Analytics\\group_project\\ml\\features_3_sec_train.csv')
data_3_train


# In[ ]:


data_3_1 = data_3_train.iloc[:,2:]
data_3_1.shape


# In[ ]:


data_3_1.groupby(data_3_1['label']).describe()


# In[ ]:


Y_3_train = data_3_1['label']
# 将除了label外的特征作为X
X_3_train = data_3_1.iloc[:,:-1]
# 将label列的文字转化成数字
map_label = {"blues":0, "classical":1, "country":2, "disco":3, "hiphop":4, "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9}
# map函数进行映射，对label中的每一个元素进行map_label中的替换
Y_3_train = Y_3_train.map(lambda x: map_label[x])
classes=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
# 查看训练集中的各类别分布，确定样本是否不均衡
print(Y_3_train.value_counts())


# In[ ]:


data_3_test = pd.read_csv('D:\\79886\\Documents\\HKUST\\Courses\\5001 Foundations of Data Analytics\\group_project\\ml\\features_3_sec_test.csv')
data_3_test


# In[ ]:


data_3_2 = data_3_test.iloc[:,2:]
data_3_2.shape


# In[ ]:


data_3_2.groupby(data_3_2['label']).describe()


# In[ ]:


Y_3_test = data_3_2['label']
# 将除了label外的特征作为X
X_3_test = data_3_2.iloc[:,:-1]
# 将label列的文字转化成数字
map_label = {"blues":0, "classical":1, "country":2, "disco":3, "hiphop":4, "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9}
# map函数进行映射，对label中的每一个元素进行map_label中的替换
Y_3_test = Y_3_test.map(lambda x: map_label[x])
classes=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
# 查看训练集中的各类别分布，确定样本是否不均衡
print(Y_3_test.value_counts())

# In[15]:


scaler = preprocessing.StandardScaler().fit(X_3_train)

# if normalization
X_3_train = scaler.transform(X_3_train)
X_3_test = scaler.transform(X_3_test)


# In[21]:

# gs = XGBClassifier(n_estimators=500,learning_rate=0.1,max_depth=5).fit(X_3_train,Y_3_train) #0.58
# gs = SVC().fit(X_3_train,Y_3_train) #0.59
gs = LogisticRegression(max_iter=1000).fit(X_3_train,Y_3_train) #0.6
# gs = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(256,128,64,10), max_iter=400, learning_rate_init=0.005).fit(X_3_train,Y_3_train) #0.568
gs.score(X_3_train,Y_3_train),gs.score(X_3_test,Y_3_test)

# In[ ]:

pred = gs.predict(X_3_test)
print(pred)

# In[ ]:

df = pd.DataFrame({'name': data_3_test['filename'], 'label': Y_3_test, 'pred':pred})
df.to_csv('D:\\79886\\Documents\\HKUST\\Courses\\5001 Foundations of Data Analytics\\group_project\\ml\\pred.csv')


# In[ ]:
from sklearn import metrics
metrics.accuracy_score(Y_3_test, pred)

# In[ ]:

c = [0]*250
lp = [0]*250
l = [0]*250
for i in range(250):
    c[i] = Counter(df.loc[10*i:10*i+10,'pred'])
    lp[i] = c[i].most_common(1)
    l[i] = lp[i][0][0]


# In[ ]:

t = [0]*250
for i in range(250):
    t[i] = df.loc[10*i,'label']
    
pred30 = pd.DataFrame({'label': t, 'pred':l})

# In[ ]:

metrics.accuracy_score(pred30['label'], pred30['pred'])

k = 0
for i in range(250):
    if pred30.loc[i,'label'] == pred30.loc[i, 'pred']:
        k+=1
        
acu = k/250
print(acu)

# In[22]:

# 概率求和
p = [0]*250
prob = gs.predict_proba(X_3_test)
for i in range(250):
    for j in range(10):
        p[i] += prob[10*i+j]


# In[22]:

p_max = [0]*250
for i in range(250):
    p[i]=p[i].tolist()
    p_max[i] = p[i].index(max(p[i]))

# In[22]:

prob30 = pd.DataFrame({'label': t, 'pred':p_max})
metrics.accuracy_score(prob30['label'], prob30['pred'])
# lr 0.592 
# xgboost 0.576
