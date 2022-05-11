#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from pickle import dump
from pickle import load
import pickle


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


st.title('FRAUD DETECTION OF HOSPITAL DATA')
st.sidebar.header('User Input Parameters')


# In[4]:


st.subheader("MODEL DEPLOYMENT: RANDOM FOREST CLASSIFIER")


# In[37]:


st.image('fraud_detection.JPG',width=8)


# In[6]:


loaded_model=pickle.load(open('f_test_data.save','rb'))


# In[7]:


def Fraud_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if(prediction[0]==0):
        return'The calim is fraud'
    else:
        return 'The calim is genuinine'


# In[8]:


def user_input_features():
    Area_Service=st.sidebar.selectbox('Area_Service',('1','2','3','4','5', '6','7'))
    Age = st.sidebar.number_input ('Enter age')
    Gender=st.sidebar.selectbox('Gender-- Male=0,Female=1,Unknown=2',('0','1','2'))
    Cultural_group=st.sidebar.selectbox('Cultural_group- White=1,Black/AfricanAmerican=2,Other Race=3,unknown=4',('1','2','3','4'))
    ethinicity=st.sidebar.selectbox('ethnicity--Not Span/Hispanic=1,Spanish/Hispanic=2,unknown=3',('1 ','2','3'))
    Days_spend_hsptl=st.sidebar.number_input("enter number of days spend in hospital")
    Admission_type=st.sidebar.selectbox('Admission_type--Emergency=1,Elective=2,Urgent=3,Newborn=4,Trauma=5,Not Available=6',('1','2','3','4','5','6'))
    home_self_care=st.sidebar.selectbox('home_self_care',('1','2','3','4','5','6'))
    ccs_diagnosis_code=st.sidebar.number_input("enter the diagnosis_code")
    ccs_procedure_code=st.sidebar.number_input("enter the procedure_code")
    Code_illness=st.sidebar.selectbox('Code_illness',('0','1','2','3','4'))
    Mortality_risk=st.sidebar.selectbox('Mortality_risk',('1','2','3'))
    Surg_Description=st.sidebar.selectbox('Surg_Description--Medical=1,Surgical=2,Non=applicabel=3',('1','2','3'))
    Weight_baby=st.sidebar.number_input("enter the weight of the baby")
    Abortion=st.sidebar.selectbox('Abortion',("1","0"))
    emergency_dept_yes = st.sidebar.selectbox('emergency_dept_yes/no--Yes=1,No=0',('1','0'))
    Tot_charg=st.sidebar.number_input("enter the charge")
    Tot_cost=st.sidebar.number_input('enter the cost')
    ratio_of_total_costs_to_total_charges=st.slider('Slide me',min_value=0.1,max_value=0.95)
    Payment_Typology=st.sidebar.number_input("enter the Payment_Typology")
    Fraud_prediction={'Area_Service':Area_Service,
           'Age':Age,
           'Gender':Gender,
           'Cultural_group':Cultural_group,
           'ethinicity':ethinicity,
           'Days_spend_hsptl':Days_spend_hsptl,
           'Admission_type':Admission_type,
           'home_self_care':home_self_care,
           'ccs_diagnosis_code':ccs_diagnosis_code,
           'ccs_procedure_code':ccs_procedure_code,
           'Code_illness':Code_illness,
           'Mortality_risk':Mortality_risk,
           'Surg_Description':Surg_Description,
           'Weight_baby':Weight_baby,
           'Abortion':Abortion,
           'emergency_dept_yes':emergency_dept_yes,
           'Tot_charg':Tot_charg,
           'Tot_cost':Tot_cost,
           'ratio_of_total_costs_to_total_charges':ratio_of_total_costs_to_total_charges,
           'Payment_Typology':Payment_Typology}
    features = pd.DataFrame(Fraud_prediction,index = [0])
    return features

        
        
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)







# In[9]:


#st.subheader('Input parameters')
#dfa=pd.read_csv('deploy1.csv')
#st.write(dfa)


# In[10]:


import pickle
loaded_model=pickle.load(open('f_test_data.save','rb'))
loaded_model


# In[11]:


f_data1=pd.read_csv('final_data.csv')


# In[12]:


prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)


# In[34]:


prediction_proba


# In[35]:


st.subheader('Prediction Probability')
prediction_proba=pd.DataFrame(prediction_proba)
b=[0]
a= prediction_proba.iloc[:,0]
if (a>0.5).any():
  a='Claim id geunine'
  #print('Claim is geunine')
else:
   b='Claim is fraud'
  # print('claim is fraud')

st.write(prediction_proba)
st.write(b)
st.write(a)


# In[36]:


output=pd.concat([df,prediction_proba],axis=1)


# In[16]:


#output


# In[ ]:





# In[ ]:



