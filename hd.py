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

plt.show()

# subject and examination variables
sub = pd.melt(hd, id_vars = ['cardio'], value_vars=['alco','smoke','active','gluc', 'cholesterol'])
g = sns.catplot(x='variable', hue = 'value', col='cardio', kind ='count',data = sub)

plt.show()

# Creating bp_level variable
hd.loc[(hd['ap_hi'] < 120) & (hd['ap_lo'] < 80), 'bp_level'] = 1 # 1 = normal
hd.loc[(hd['ap_hi'] >= 120) & (hd['ap_hi'] < 130) & (hd['ap_lo'] < 80), 'bp_level'] = 2 # 2 = elevated
hd.loc[(hd['ap_hi'] >= 130) & (hd['ap_hi'] < 140) | (hd['ap_lo'] >= 80) & (hd['ap_lo'] < 90),'bp_level'] = 3 # 3 = 'high bp 1'
hd.loc[(hd['ap_hi'] >= 140) & (hd['ap_hi'] < 180) | (hd['ap_lo'] >= 90) & (hd['ap_lo'] <120 ), 'bp_level'] = 4 # 4 ='high bp 2'
hd.loc[(hd['ap_hi'] >= 180) | (hd['ap_lo'] >= 120) , 'bp_level'] = 5 # 5 = 'hypertensive crisis'



hd = hd.drop(['ap_hi', 'ap_lo'], axis = 1)

# bp_level and cardio
bp = sns.countplot(x='bp_level', hue = 'cardio', data= hd)

print(hd.head())


# BMI
hd['bmi'] = hd['weight']/((hd['height']/100) ** 2)

# Clean the outliers which is below 2.5% and above 97.5% of the data.
hd.drop(hd[(hd['bmi'] > hd['bmi'].quantile(0.975)) | (hd['bmi'] < hd['bmi'].quantile(0.025))].index, inplace=True)

print(hd.bmi.describe())

bmi = sns.FacetGrid(hd, col = 'cardio', hue = 'gender')
bmi.map(plt.scatter, 'bmi', 'age' ).add_legend()

plt.show()

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

with sns.axes_style("dark"):
    g = sns.FacetGrid(hd, hue="cardio", col="cardio", height=4)
g.map(hexbin, "age", "bmi", extent=[20,100,0, 50])

plt.show()

# Correlation

f, ax= plt.subplots(figsize=(11,9))
cols = hd.corr().nlargest(13, 'cardio')['cardio'].index
cm = np.corrcoef(hd[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

# Split train and test data
x = hd.drop("cardio", axis=1)
y = hd["cardio"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# Support Vector Machines
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(x_train, y_train)
Y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_test, y_test) * 100,2)

# Naive bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
Y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_test, y_test) * 100, 2)


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_test)
random_forest.score(x_test, y_test)
acc_random_forest = round(random_forest.score(x_test, y_test) * 100, 2)
print(acc_random_forest)

# KNN Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_test, y_test) * 100,2)


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dc_tree = DecisionTreeClassifier()
dc_tree.fit(x_train, y_train)
y_pred = dc_tree.predict(x_test)
acc_dc_tree = round(dc_tree.score(x_test, y_test) * 100 ,2)

# Xgboost Classifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xg = XGBClassifier()
xg.fit(x_train, y_train)
y_pred = xg.predict(x_test)
acc_xg = round(xg.score(x_test, y_test) * 100, 2)




# Find the accuarcy
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 
              'Random Forest','Gaussian', 'Decision Tree', 'XgBoost'],
    'Score': [acc_svc, acc_knn, acc_random_forest, 
             acc_gaussian, acc_dc_tree, acc_xg]})
models.sort_values(by='Score', ascending=False)

print(models)

# classification report and accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = xg.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


