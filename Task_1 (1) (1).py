#!/usr/bin/env python
# coding: utf-8

# BY - KHADIJAH SHAMWIL

# #### TASK-1 Prediction Using Supervised Machine Learning

# #### Perform exploratory data analysis on dataset 'Student' to predict the percentage of marks of the students based on the number of hours they studied

# ### Dataset sample: http://bit.ly/w-data

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')


# In[2]:


# Importing the required libraries 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 


# ### Reading the data

# In[3]:


data = pd.read_csv('http://bit.ly/w-data') 
data.head(10)


# In[4]:


# check if there is any null value in the Dataset
data.isnull == True


# #### There is no null value in the Dataset so, we can now visualize our Data.
# 

# In[5]:


sns.set_style('darkgrid') 
sns.scatterplot(y = data['Scores'], x= data['Hours']) 
plt.title('Marks Vs study Hours' , size=20)
plt.ylabel('Marks Percentage' , size=12)
plt.xlabel('Hours studied' , size=12)
plt.show()


# #### From the above scatter plot there looks to be correlation between the MARKS PERCENTAGE AND THE HOURS STUDIED, Lets plot a regression line to confirm the correlation.

# In[6]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression plot' ,size= 20)
plt.ylabel('Marks Percentage' , size=12)
plt.xlabel('Hours Studied' , size=12)
plt.show()
print(data.corr())


# #### It is confirmed that the variable are positively correlated.

# #### Training the model
# 

# #### 1) Splitting the data

# In[7]:


# Defining X and y from the data

X = data.iloc[:,:-1].values
y = data.iloc[:, 1].values

#splitting the data into two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# #### 2) Fitting the data into the model

# In[8]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("----------Model Trained------------")


# #### Predicting the percentage of marks

# In[9]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks' : [k for k in pred_y]})
prediction


# #### Comparing the Predicted Marks with the actual marks

# In[10]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# #### Visually comparing the Predicted Marks with the actual Marks

# In[11]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title ('Actual vs Predicted', size = 20)
plt.ylabel('Marks Percentage', size = 12)
plt.xlabel('Hours Studied', size = 12)
plt.show()


# #### Evaluating the Model

# In[12]:


# calculating the accuracy of the model
print ('Mean Absolute Error :', mean_absolute_error(val_y, pred_y))


# #### Small value of mean absolue error states that the chances of error or wrong forecasting through the model are very less

# #### What will be the predicted score of the student if he/she studies for 9.25 hrs/day?

# In[13]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# #### According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks

# In[ ]:




