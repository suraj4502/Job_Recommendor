#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os


# In[3]:


os.getcwd()
os.chdir("C:\\Users\\RAJESH\\Desktop")
Data = pd.read_csv("Final_Train_Dataset.csv")


# In[4]:


Data.head(6)


# In[52]:


Data.info()


# In[5]:


print(Data['job_desig'])
print(Data['job_desig'].isnull())


# In[6]:


print(Data['job_description'])
print(Data['job_description'].isnull())


# In[7]:


print(Data.isnull().values.any())


# In[8]:


print(Data.isnull().sum().sum())


# In[9]:


D1=Data.drop(['job_type'],axis=1)


# In[10]:


print(D1.isnull().sum().sum())


# In[11]:


missing = D1.columns[D1.isnull().any()]


# In[12]:


print(missing)


# In[13]:


D1.isna().sum()


# In[14]:


D2=D1.dropna( subset=["job_description", "key_skills"], inplace=True)
D2=D1.shape
print(D2)


# In[15]:


D2=D1.head(15384)


# In[54]:


print(D2)


# In[17]:


D2.shape


# In[18]:


D2["company_name_encoded"]=D2["company_name_encoded"].astype("str")


# In[19]:


D2['comb']= D2[['experience','job_description','job_desig','key_skills','location','salary','company_name_encoded']].apply(lambda x: ' '.join(x), axis=1)


# In[20]:


D2.head()


# In[21]:


Enter=input()


# In[22]:


Enter


# In[23]:


import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


nltk.download('stopwords')


# In[26]:


# Program to measure the similarity between 
# two sentences using cosine similarity.
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[27]:


def nltk_cosine(input1, input2):
    if (input1 is not None) & (input2 is not None):
        X_list = word_tokenize(input1) 
        Y_list = word_tokenize(input2)
        sw = stopwords.words('english') 
        l1 =[];l2 =[]
        X_set = {w for w in X_list if not w.lower() in sw} 
        Y_set = {w for w in Y_list if not w.lower() in sw}
        rvector = X_set.union(Y_set) 
        for w in rvector:
            if w in X_set: l1.append(1) 
            else: l1.append(0)
            if w in Y_set: l2.append(1)
            else: l2.append(0)
        c = 0
        for i in range(len(rvector)):
            c+= l1[i]*l2[i]
            cosine = c / float((sum(l1)*sum(l2))**0.5)
    else:
        cosine=0
    return(cosine)


# In[28]:


D2["Score"]=D2["comb"].apply(lambda x: nltk_cosine(x, Enter))


# In[29]:


D3=D2.head(15384)


# In[30]:


D3


# In[31]:


df=pd.DataFrame(D3,columns=['Score','comb'])


# In[32]:


df.sort_values(by=['Score'],ascending=False)


# In[37]:


get_ipython().system('pip install --user streamlit')
import streamlit as st


# In[38]:


import pickle
import numpy as np


# In[55]:


st.title("Your Data Sci Job")
job_df=pickle.load(open("Enter.pkl","rb"))
similarity=pickle.load(open("df.pkl","rb"))
list_job=np.array(job_df["comb"])
option = st.selectbox(
"Search Job",
(list_job))


# In[42]:


np.array(df["comb"])


# In[56]:


df1=pickle.load(open("Enter.pkl","rb"))
similarity=pickle.load(open("df.pkl","rb"))
list_job=np.array(df["comb"])
option = st.selectbox(
"Search Job ",
(list_job))


# In[58]:


def jobs_recommend(jobs):
     index = df1[df1['comb'] == jobs].index[0]
     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
     l=[]
     for i in distances[1:6]:
          l.append("{}".format(df1.iloc[i[0]].comb))
          # return("{} {}".format(movie_df.iloc[i[0]].title, movie_df.iloc[i[0]].urls))
     return(l)
if st.button('Recommend Me'):
     st.write('Jobs for you are:')
     # st.write(movie_recommend(option),show_url(option))
     df1 = pd.DataFrame({
          'Jobs Recommended': jobs_recommend(option)
     })

     st.table(df)


# In[ ]:




