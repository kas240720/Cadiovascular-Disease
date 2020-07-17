import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt # for plot


hd = pd.read_csv("C:/Users/sec/Desktop/Career/Kaggle Project/Heart Disease/heart disease.csv")

print(hd.shape)
print(hd.info)
print(hd.head())


# Check if there is any missing data
total = hd.isnull().sum().sort_values(ascending =False)
percent = (hd.isnull().sum()/hd.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
print(missing_data)

print(hd.describe)
