#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#get_ipython().run_line_magic('matplotlib', 'inline')
import sys


# In[6]:


dataset = pd.read_csv(sys.argv[1])


# In[7]:


dataset.shape


# In[8]:


dataset.describe()


# In[9]:


dataset.plot(x='x', y='y', style='o')  
plt.title('Linear Model')  
plt.xlabel('X')  
plt.ylabel('Y')  
plt.savefig("py_orig.png")


# In[21]:


X = dataset['x'].values.reshape(-1,1)
y = dataset['y'].values.reshape(-1,1)


# In[37]:


X_train = X
X_test = X
y_train = y
y_test = y


# In[38]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[39]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[40]:


y_pred = regressor.predict(X_test)


# In[41]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[42]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[43]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.savefig("py_lm.png")


# In[ ]:




