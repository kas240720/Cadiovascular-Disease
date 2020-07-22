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

## EDA

# age
hd['age'] = (hd['age']/365).round().astype('int')
# ag = sns.countplot(x = 'age', hue='cardio' ,data=hd)


# subject and examination variables
sub = pd.melt(hd, id_vars = ['cardio'], value_vars=['alco','smoke','active','gluc', 'cholesterol'])
# g = sns.catplot(x='variable', hue = 'value', col='cardio', kind ='count',data = sub)

# Creating bp_level variable
hd.loc[(hd['ap_hi'] < 120) & (hd['ap_lo'] < 80), 'bp_level'] = 'normal'
hd.loc[(hd['ap_hi'] >= 120) & (hd['ap_hi'] < 130) & (hd['ap_lo'] < 80), 'bp_level'] = 'elevated'
hd.loc[(hd['ap_hi'] >= 130) & (hd['ap_hi'] < 140) | (hd['ap_lo'] >= 80) & (hd['ap_lo'] < 90),'bp_level'] = 'high bp 1'
hd.loc[(hd['ap_hi'] >= 140) & (hd['ap_hi'] < 180) | (hd['ap_lo'] >= 90) & (hd['ap_lo'] <120 ), 'bp_level'] = 'high bp 2'
hd.loc[(hd['ap_hi'] >= 180) | (hd['ap_lo'] >= 120) , 'bp_level'] = 'hypertensive crisis'

hd = hd.drop(['ap_hi', 'ap_lo'], axis = 1)

# bp_level and cardio
bp = sns.countplot(x='bp_level', hue = 'cardio', data= hd)

print(hd.head())


# BMI
hd['bmi'] = hd['weight']/((hd['height']/100) ** 2)

# Clean the outliers which is below 2.5% and above 97.5% of the data.
hd.drop(hd[(hd['bmi'] > hd['bmi'].quantile(0.975)) | (hd['bmi'] < hd['bmi'].quantile(0.025))].index, inplace=True)

print(hd.bmi.describe())

# bmi = sns.FacetGrid(hd, col = 'cardio', hue = 'gender')
# bmi.map(plt.scatter, 'bmi', 'age' ).add_legend()

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

with sns.axes_style("dark"):
    g = sns.FacetGrid(hd, hue="cardio", col="cardio", height=4)
g.map(hexbin, "age", "bmi", extent=[20,100,0, 50]);


# Correlation

f, ax= plt.subplots(figsize=(11,9))
cols = hd.corr().nlargest(13, 'cardio')['cardio'].index
cm = np.corrcoef(hd[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# split train and test data

# Decisiontree classfier, randomforest, machine learning

#find the accuarcy



