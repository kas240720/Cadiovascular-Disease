# GOAL : 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt # for plot


hd = pd.read_csv("C:/Users/sec/Desktop/Career/Kaggle Project/Cardiovascular-Disease/cardio_train.csv", sep = ";")

print(hd.shape)
print(hd.info())
print(hd.head())


# Check if there is any missing data
total = hd.isnull().sum().sort_values(ascending =False)
percent = (hd.isnull().sum()/hd.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
print(missing_data)

# EDA

# age
hd['age'] = (hd['age']/365).round().astype('int')
ag = sns.countplot(x = 'age', hue='cardio' ,data=hd)


# subject and examination variables
sub = pd.melt(hd, id_vars = ['cardio'], value_vars=['alco','smoke','active','gluc', 'cholesterol'])
g = sns.catplot(x='variable', hue = 'value', col='cardio', kind ='count',data = sub)

# Creating bp_level variable
hd.loc[(hd['ap_hi'] < 120) & (hd['ap_lo'] < 80), 'bp_level'] = 'normal'
hd.loc[(hd['ap_hi'] >= 120) & (hd['ap_hi'] < 130) & (hd['ap_lo'] < 80), 'bp_level'] = 'elevated'
hd.loc[(hd['ap_hi'] >= 130) & (hd['ap_hi'] < 140) | (hd['ap_lo'] >= 80) & (hd['ap_lo'] < 90),'bp_level'] = 'high blood pressure 1'
hd.loc[(hd['ap_hi'] >= 140) & (hd['ap_hi'] < 180) | (hd['ap_lo'] >= 90) & (hd['ap_lo'] <120 ), 'bp_level'] = 'high blood pressure 2'
hd.loc[(hd['ap_hi'] >= 180) | (hd['ap_lo'] >= 120) , 'bp_level'] = 'hypertensive crisis'

hd = hd.drop(['ap_hi', 'ap_lo'], axis = 1)

# bp_level and cardio
bp = sns.countplot(x='bp_level', hue = 'cardio', data= hd)
plt.show()

