import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Job Recommendor System")
df=pickle.load(open("movie_recm.pkl","rb"))
similarity=pickle.load(open("similarity.pkl","rb"))


ls=np.array(df["comb"])

option = st.text_input("Enter Candidate Skills : ")

def recommend(skills):
    index = df[df['key_skills'] == skills].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    jd = []
    ks = []
    loc = []
    sal = []
    for i in distances[1:8]:
        jd.append(df.iloc[i[0]].job_description)
        ks.append(df.iloc[i[0]].key_skills)
        loc.append(df.iloc[i[0]].location)
        sal.append(df.iloc[i[0]].salary)
    return jd, ks, loc, sal

if st.button('Recommend Me'):
     st.write('Job Recommendations for you are:')
     # st.write(recommend(option))
     jd, ks, loc, sal = recommend(option)

     opd = pd.DataFrame()
     opd['job_description']= jd
     opd['key_skills']= ks
     opd['location']= loc
     opd['salary']= sal


     st.dataframe(opd)