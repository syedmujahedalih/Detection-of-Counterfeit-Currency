# -*- coding: utf-8 -*-
######################################################################
#EEE 591 : Python for Rapid Engineering Solutions, Fall 2020         #
#Project 1 - Part A                                                  #
#Analyzing the Dataset provided by AMAPE Corporation                 #
#Author: Mujahed Syed, ASU ID: 1217635874                            #
#Instructor: Dr. Steven Millman                                      #
######################################################################

#Importing the required packages

import numpy as np                #For statistics and linear algebra 
import pandas as pd               #For reading in the input data
import matplotlib.pyplot as plt         #For plotting
import seaborn as sns                   #For plotting

#Importing the input data in a Pandas DataFrame

df = pd.read_csv("data_banknote_authentication.txt")          #Using pandas built in method to import csv or text data

#Printing an empty line for better readability on the console
print(" ")


#Finding Correlation

corr_mat = df.corr().abs()             #Returns the absolute values of correlation of each variable with another as a correlation matrix
sns.heatmap(corr_mat,cbar=True,annot=True,square=True)            #Plotting the correlation matrix as a heatmap using seaborn                 
plt.title('Correlation Matrix')                                   #Giving the heatmap a title 
plt.show()                                                        #Displaying the heatmap
corr_mat *= np.tri(*corr_mat.values.shape, k=-1).T                #Transpose of triangular matrix and element by element multiplication with correlation matrix
corr_unstack = corr_mat.unstack()                                 #Unstacking the modified correlation matrix
corr_unstack.sort_values(inplace=True,ascending=False)            #Sorting the unstacked correlation matrix in descending order
print("Variables Highly Correlated with Each Other: ")
print(" ")                                                        #For better readability
print(corr_unstack.head(5))   #Displaying Variables highly correlated with each other
print(" ")                                                        #For better readability
with_class = corr_unstack.get(key='class')               #How other variables correlate with "class" label
print("How other Variables Correlate with \"Class\" label: ")         
print(" ")                                                        #For better readability 
print(with_class)
print(" ")                                                        #For better readability 

#Finding Covariation

covar = df.cov().abs()                      #Returns the absolute values of covariance of each variable with another as a covariance matrix


#Heat Map Covariance matrix

sns.heatmap(covar,cbar=True,annot=True,square=True)              #Plotting the covariance matrix as a heatmap using seaborn
plt.title('Covariance Matrix')                                   #Giving the heatmap a title
plt.show()                                                       #Displaying the heatmap
covar *= np.tri(*covar.values.shape, k=-1).T                     #Transpose of triangular matrix and element by element multiplication with covariance matrix
covar_unstack = covar.unstack()                                  #Unstacking the modified covariance matrix
covar_unstack.sort_values(inplace=True,ascending=False)          #Sorting the unstacked covariance matrix in descending order 
print("Variables Having High Covariance: ")      
print(" ")                                                       #For better readability
print(covar_unstack.head(5))                                     #Printing variables that have high covariance with each other
print(" ")
with_class_covar = covar_unstack.get(key="class")                #Covariance of other variables with the class label
print("Covariance of other variables with the \"Class\" label: ")
print(" ")                                                       #For better readability 
print(with_class_covar)                          





#Pair Plot
  
sns.set(style='whitegrid', context='notebook')                 #setting the style for the pair plot
g = sns.pairplot(df,height=1.5,hue='class',diag_kind='hist')   #Creating the pair plot and distinguishing class labels using the 'hue' argument 
g.fig.suptitle('Pair Plot',y=1.02)                             #Titling the pair plot
plt.show()                                                     #Displaying the pair plot

