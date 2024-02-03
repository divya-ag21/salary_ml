#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# In[11]:


df = pd.read_csv('salaries.csv')
df1=pd.read_csv('salaries.csv')
df


# In[14]:


plt.scatter(df['degree'],df['salary_more_then_100k'],marker='+',color='red')


# In[15]:


X = df.drop(columns=['salary_more_then_100k'])
Y = df['salary_more_then_100k']


# In[16]:


X=pd.get_dummies(X,dtype=int)
X


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.15)


# In[18]:


model = LogisticRegression()
model.fit(X,Y)


# In[19]:


ans=model.predict(x_test)
ans


# In[20]:


predic=model.predict([[1,0,0,1,0,0,0,1]])
predic


# In[ ]:


model.score(x_train,y_train)

