# PROJECT2
# COMP450
# Esra GENECİ
# Eyüp YASUNTİMUR
# 16.12.2019
import pandas as pd
import matplotlib.pyplot as plt
import scipy,numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm

data = pd.read_csv ("dataset.csv", sep=";")
array=data.values
print(data,"\n\n")
x=array[:,0:7]
y=array[:,6]

############################################################################################################################
##### PART A ##### PART A ##### PART A ##### PART A ##### PART A ##### PART A ##### PART A ##### PART A ##### PART A
############################################################################################################################

## Describe the Data
print("\n\n", data.info())
print("\n\n", data.dtypes)
print("\n\nData Describe:\n", data.describe(),"\n\n")
print("Data Shape:\n",data.shape,"\n\n")

## Grouping the Data
print("Groupby:",data.groupby('R').size(),"\n")
print("Groupby:",data.groupby('L').size(),"\n")
print("Groupby:",data.groupby('VAS').size(),"\n")
print("Groupby:",data.groupby('AR').size(),"\n")
print("Groupby:",data.groupby('HD').size(),"\n")

## Histogram
data.hist(bins=15)
plt.show()

## Density Plots
data.plot(kind='density',subplots=True,sharex=False)
plt.show()

## Box and Whisker Plots
data.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()

##Correlation Matrix Plot
correlations=data.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=0,vmax=1)
fig.colorbar(cax)
ticks=numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()

## Plotting of Train and Test Set in Python
y=data.R
x=data.drop('R',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=lm().fit(x_train,y_train)
predictions=model.predict(x_test)
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()

plt.hist(data.R, bins=30)
plt.title('R histogram')
plt.xlabel('value')
plt.ylabel('count')
plt.show()

plt.hist(data.L, bins=50)
plt.title('L histogram')
plt.xlabel('value')
plt.ylabel('count')
plt.show()

plt.hist(data.VAS, bins=30)
plt.title('VAS histogram')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

############################################################################################################################
##### PART B ##### PART B ##### PART B ##### PART B ##### PART B ##### PART B ##### PART B ##### PART B
############################################################################################################################


## Scale Process of Data ## Rescaling Data
scaler = MinMaxScaler(feature_range=(0,1))
scaledData=scaler.fit_transform(x)
numpy.set_printoptions(precision=3) #Setting precision for the output
print("Scale (0,1): \n",scaledData,"\n\n")

## Standardizing Data
scaler =StandardScaler().fit(x)
scaledData=scaler.transform(x)
print("Standardizing Data (0,1): \n",scaledData,"\n\n")

## Normalizing Data
scaler=Normalizer().fit(x)
normalizedData=scaler.transform(x)
print("Normalized Data: \n",normalizedData,"\n\n")

############################################################################################################################
##### PART C ##### PART C ##### PART C ##### PART C ##### PART C ##### PART C ##### PART C ##### PART C ##### PART C ##### PART C ##### PART C
############################################################################################################################

##A Method that is scaled and normalized the data
def scale(dataa):
    i = 0
    min_value=dataa[i+1]
    for i in range(0,len(dataa)-1):
        if(dataa[i]<min_value):
            min_value=dataa[i]
    j=0
    max_value=dataa[j+1]
    for j in range(0,len(dataa)-1):
        if(dataa[j]>max_value):
            max_value=dataa[j]
    normalized=(dataa-min_value)/(max_value-min_value)
    return normalized

data = pd.read_csv ("dataset.csv", sep=";")
x=data.drop(['ID'], axis=1)
data['R']=scale(data['R'])
data['L']=scale(data['L'])
data['BT']=scale(data['BT'])
data['VAS']=scale(data['VAS'])
data['AR']=scale(data['AR'])
data['HD']=scale(data['HD'])

numpy.set_printoptions(precision=3)
x=data
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
print(x_pca)
print("Variance Ratio: ", pca.explained_variance_ratio_)
print("Sum Variance: ", sum(pca.explained_variance_ratio_))
data['P1']=x_pca[:, 0]
data['P2']=x_pca[:, 1]
target_names=["hernia","not hernia"]
color= ["red","green"]
for each in range(2):
    plt.scatter(data.P1[data.HD==each], data.P2[data.HD==each], color=color[each], label=target_names[each])

plt.legend()
plt.show()


################################################################################################################################################################################################################################################
##### PART D ##### PART D ##### PART D ##### PART D ##### PART D ##### PART D ##### PART D ##### PART D ##### PART D ##### PART D
######################################################################################################################################################################
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##A Method that is scaled and normalized the data
def scale(dataa):
    i = 0
    min_value=dataa[i+1]
    for i in range(0,len(dataa)-1):
        if(dataa[i]<min_value):
            min_value=dataa[i]
    j=0
    max_value=dataa[j+1]
    for j in range(0,len(dataa)-1):
        if(dataa[j]>max_value):
            max_value=dataa[j]
    normalized=(dataa-min_value)/(max_value-min_value)
    return normalized

dataa = pd.read_csv ("dataset.csv",sep=";")
dataa.head()
number=LabelEncoder()
dataa['R']=scale(dataa['R'])
dataa['L']=scale(dataa['L'])
dataa['BT']=scale(dataa['BT'])
dataa['VAS']=scale(dataa['VAS'])
dataa['AR']=scale(dataa['AR'])
dataa['HD']=scale(dataa['HD'])
features = ["R", "L", "BT", "VAS","AR"]
target="HD"
features_train, features_test, target_train, target_test = train_test_split(dataa[features],dataa[target],test_size = 0.33,random_state = 54)

##GaussianNB
model = GaussianNB()
model.fit(features_train, target_train)
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(" Acurracy score for GNB: ",accuracy)

##Dessicion Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, target_train)
pred2=clf.predict(features_test)
accuracy2=accuracy_score(target_test, pred2)
print(" Acurracy score for Desicion Tree: ",accuracy2)

##Linear SVM
from sklearn.svm import LinearSVC
cld= LinearSVC(random_state=0, tol=1e-5, C=0.001)
cld.fit(features_train, target_train)
pred3=cld.predict(features_test)
accuracy3=accuracy_score(target_test, pred3)
print(" Acurracy score for Linear SVM: ",accuracy3)
