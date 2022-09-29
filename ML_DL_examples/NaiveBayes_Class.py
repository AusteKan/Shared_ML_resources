"""
Nayve Bayes implementation using iris dataset as an example.

The code is structured as a class and can be integrated into other workflows.
"""


#%%--------------------------------------
#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


#%%--------------------------------------
class NayveBayes():

    """
    Nayve Bayes assessment using Gaussian probabilistic modelling
    """
    def __init__(self):
        pass
    def group_data(self,X_train, y_train, classes):
        #Get data mean, STD, and counts per class and feature
        #Function returns  mean_matrix,std_matrix,priors_matrix where 
        #rows are classes and columns are features

        n=X_train.shape[1] #get the number of features
        n_samples=X_train.shape[0] #number of samples
        #Mean matrix
        mean_matrix=np.zeros((len(classes),n))
        #STD matrix
        std_matrix=np.zeros((len(classes),n))
        #Get prior probabilities
        priors_matrix=np.zeros((len(classes),n))
    
        for i,class_val in enumerate(classes):
            idx=np.where(y_train==class_val)
            for j in range(n):
                subset=X_train[idx,j]
                mean_matrix[i,j]=np.mean(subset)
                std_matrix[i,j]=np.std(subset)
                priors_matrix[i,j]=len(subset[0])/n_samples
        
        return mean_matrix,std_matrix,priors_matrix
    
    def Gaussian(self,mean,std,x):

        #Get gaussian probability for a specific value or array of values

        return np.exp((-(x-mean)**2)/(2*std**2))*(1/(np.sqrt(np.pi*2)*std))

    def fit(self,X_train,y_train):
        
        #Fit data 
        self.classes=np.unique(y_train)
        self.mean_matrix,self.std_matrix,self.priors_matrix=self.group_data(X_train, y_train, self.classes)

    def get_prob(self,sample):
        #The function returns class probabilities
        probabilities=[]
        for i in range(len(self.classes)):
            mean=self.mean_matrix[i]
            std=self.std_matrix[i]
            posterior=np.sum(np.log(self.Gaussian(mean,std,sample)))
            priors=np.sum(np.log(self.priors_matrix[i]))
            prob=posterior+priors
            probabilities.append(prob)
        return probabilities

    def predict(self,X_test):
        #Returns probability list with a predicted probability for each class
        pred_list=[]

        for i in range(X_test.shape[0]):

            sample=X_test[i]
            prob=self.get_prob(sample)
            pred_list.append(prob)

        return pred_list


#%%--------------------------------------
#Import and prepare test data

data_iris=load_iris()

data=pd.DataFrame(data=data_iris.data, columns=data_iris.feature_names)
target=data_iris.target

print(data.head())
print(target[1:10])


X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#%%--------------------------------------
#Test Naive Bayes
nb=NayveBayes()
nb.fit(X_train,y_train)

y_pred=nb.predict(X_test)

y_pred_vals=np.argmax(y_pred,axis=1)
#%%--------------------------------------
#Evaluate the model

print(classification_report(y_pred_vals,y_test))

sns.heatmap(confusion_matrix(y_pred_vals,y_test),annot=True,cmap="Blues",fmt='g')



#%%--------------------------------------
#Compare to sklearn model

sk_nb=GaussianNB()
sk_nb.fit(X_train,y_train)
y_pred_sk=sk_nb.predict(X_test)

print(classification_report(y_pred_sk,y_test))

sns.heatmap(confusion_matrix(y_pred_sk,y_test),annot=True,cmap="Blues",fmt='g')

            

#%%--------------------------------------
#Create a classification dataset

X,y=make_classification(n_samples=1000,n_features=5,n_informative=3,n_redundant=2,n_classes=4)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Test Naive Bayes class
nb=NayveBayes()
nb.fit(X_train,y_train)

y_pred=nb.predict(X_test)

y_pred_vals=np.argmax(y_pred,axis=1)

print(classification_report(y_pred_vals,y_test))

sns.heatmap(confusion_matrix(y_pred_vals,y_test),annot=True,cmap="Blues",fmt='g')

# %%---------------------------------
#Compare results to sklearn Naive Bayes
sk_nb=GaussianNB()
sk_nb.fit(X_train,y_train)
y_pred_sk=sk_nb.predict(X_test)

print(classification_report(y_pred_sk,y_test))

sns.heatmap(confusion_matrix(y_pred_sk,y_test),annot=True,cmap="Blues",fmt='g')

# %%
