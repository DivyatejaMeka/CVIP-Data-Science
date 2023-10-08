#!/usr/bin/env python
# coding: utf-8

# # Phase 1 - Normal Task

# # EDA - Terrorism
# 

# ## Problem Statement :-

# ### The goal of this project is to perform Exploratory Data Analysis (EDA) on the terrorism dataset to extract meaningful insights and patterns. By analyzing the dataset, we aim to uncover trends, identify high-risk regions, understand attack characteristics, and potentially reveal factors that contribute to terrorist activities.

# ## Let's Start

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('C:/Users/Admin/Desktop/CVIP EDA PRoj/globalterrorismdb_0718dist.csv', encoding='latin1')


# In[8]:


pd.set_option("display.max_columns",500)


# In[9]:


data.head()


# ## Cleaning the data

# In[73]:


#Extracting Columns
columns = ['iyear','imonth','iday','country_txt','city','latitude','longitude',
          'location','attacktype1_txt','targtype1_txt','targetsubtype1_txt','target1',
          'gname','motive','weapontype1_txt','dbsource','region_txt','nkill','nwound',
          'natlty1_txt','weapondetail']


# In[74]:


data_clean = pd.DataFrame(data = data, columns = columns)


# In[75]:


data_clean.shape


# In[76]:


data_clean.isnull().sum()


# In[77]:


data_clean.head(5)


# In[78]:


data_clean.rename(columns = {'iyear':'year',
                            'imonth':'month',
                            'iday':'day',
                            'country_txt':'country',
                            'attacktype1_txt':'attacktype',
                            'targtype1_txt':'targetype',
                            'targetsubtype2':'targetsubtype',
                            'gname':'group name',
                            'weapontype1_txt':'weapontype',
                            'dbsource':'source',
                            'region_txt':'region',
                            'nkill':'killed',
                            'nwound':'wounded',
                            'weapondetail':'weapons_detail',
                            'natlty1_txt':'nationality'}, inplace = True)


# In[79]:


pd.set_option("display.max_columns", 500)


# In[80]:


data_clean.head(5)


# # Number of Kills from year 1970 - 2017
# 

# In[18]:


data['nkill'].fillna(0)
no_of_kills = data.groupby('iyear')['nkill'].sum()
plt.subplots(figsize=(15,10))
plot1 = sns.barplot(x=no_of_kills.index, y=no_of_kills)
plot1.set_xlabel("Year")
plot1.set_ylabel("No Of Deaths")
plot1.set_xticklabels(no_of_kills.index,rotation=90)
plt.show()


# #### This Bar graph shows the number of deaths occurred due to terrorist activities. We can see that there has been a gradual rise in terrorist attacks since 2012 where the number of deaths were over 15000 and these terrorist attacks were at peak during the year 2014 where the total amount of deaths are over 40000. 

# In[23]:


from ipywidgets import interact, interact_manual


# In[28]:


@interact
def attack_year(year = list(data['iyear'].unique())):
    a = data[data['iyear'] == year]
    return sns.countplot(y = a['region_txt'], palette = 'viridis')


# #### We built an interactive function where we can check which regions were affected the most during a particular year. From the above data we can see that Middle East and North Africa are the most affected regions during the year 2014 where the terrorism was at peak

# ## Nations that suffered more loss due to terrorism

# In[35]:


nationality_top = data_clean[data_clean['nationality'] != 'Unknown']


# In[36]:


nationality_type = nationality_top['nationality'].value_counts().reset_index()


# In[48]:


nationality_type.rename(columns = {"index":'Num', "nationality":'Nationality', "count":'Counts'}, inplace = True)
nationality_type


# In[46]:


data_clean['nationality'].unique()


# In[50]:


f, ax = plt.subplots(figsize=(15, 10))
ax = sns.barplot(x="Counts", y="Nationality", data=nationality_type[:10],
                 palette="viridis").set_title('Nations that suffered more loss due to terrorism')


# #### From the above data we can see that Iraq suffered more loss and casualities due to terrorism

# In[51]:


@interact
def country_name(region = list(data_clean['region'].unique())):
    a = data[(data_clean['region'] == region)]
    a = pd.DataFrame(a['country_txt'].value_counts())
    return a.style.background_gradient(cmap = 'plasma')


# ## Highly active groups that cause terrorism

# In[54]:


data = data_clean[data_clean['group name'] != 'Unknown']
data_group_org=data['group name'].value_counts().reset_index()
data_group_org.rename(columns={"index":'Num', "group name":'Group Name', "count":'Counts'},inplace=True)
data_group_org


# In[55]:


f, ax = plt.subplots(figsize=(15, 10))
ax = sns.barplot(x="Counts", y="Group Name", data=data_group_org[:10],
                 palette="pastel").set_title('Groups Responsible Behind the Terror Attacks')


# #### From the above data we can see that Taliban is highly active group that causes terrorism around the world.

# ## High Risk Cities Around The World

# In[56]:


data = data_clean[data_clean['city'] != 'Unknown']
data_city = data['city'].value_counts().reset_index()
data_city.rename(columns = {"index":'Num', "city":'City', "counts":'Counts'}, inplace = True)
data_city


# In[60]:


f, ax = plt.subplots(figsize=(15, 10))
ax = sns.barplot(x="count", y="City", data=data_city[:10],
                 palette="husl").set_title('Cities highly affected by terrorist groups')


# #### From the above data we can see that Baghdad is the highly affected city due to terrorism

# ## Attack Types by terrorists

# In[66]:


data = data_clean[data_clean['attacktype'] != 'Unknown']
data_attack = data['attacktype'].value_counts().reset_index()
data_attack.rename(columns={"index":'Num',"attacktype":'Attacktype', "count":'Count'}, inplace=True)
data_attack


# In[71]:


f, ax = plt.subplots(figsize=(15, 10))
ax = sns.barplot(x="Count", y="Attacktype", data=data_attack[:10],
                 palette="flare").set_title('Attack type by terrorist')


# #### Terrorists tend to use Bombing/Explosion more to commit terrorism and create havoc

# # From the above whole data visualization we are able to understand the trends, identify high-risk regions, understand attack characteristics, and potentially reveal factors that contribute to terrorist activities.

# ## We have seen that 2014 is the year when the terrorism was at it's peak. Iraq is the nation that suffered more loss due to terrorism. Taliban is the highly active group that is causing terrorism. Baghdad is the most highly active city where terrorism wreaks havoc. Terrorists mainly cause terrorism by Bombing/Explosions, Armed assaults, Assasinations etc....
# 

# ## I sincerely thank CodersCave for giving me this oppurtunity to showcase my skills and complete this project.

# In[ ]:




