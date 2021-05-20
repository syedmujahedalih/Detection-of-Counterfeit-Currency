# -*- coding: utf-8 -*-
######################################################################
#EEE 591 : Python for Rapid Engineering Solutions, Fall 2020         #
#Project 1 - Part B                                                  #
#Detecting Counterfeit Currency bills using Machine Learning         #
#Author: Mujahed Syed, ASU ID: 1217635874                            #
#Instructor: Dr. Steven Millman                                      #
######################################################################

#Importing the Required Packages

import pandas as pd                                         #For importing the data
import numpy as np                                          #For linear algebra and statistics
from sklearn.model_selection import train_test_split           # splits database
from sklearn.preprocessing import StandardScaler              # standardize data
from sklearn.linear_model import Perceptron                   #The Perceptron algorithm
from sklearn.linear_model import LogisticRegression           #The Logistic Regression Algorithm 
from sklearn.svm import SVC                                   #The SVM(SVC) classifier
from sklearn.tree import DecisionTreeClassifier               #The decision tree classifier
from sklearn.ensemble import RandomForestClassifier           #Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier            #K-Nearest Neighbors Classifier 
from sklearn.metrics import accuracy_score                    # grade the results


df = pd.read_csv("data_banknote_authentication.txt")           #Reading in the data  
X = df.iloc[:,:-1].values                                      #Separating the features from the labels 
Y = np.ravel(df.iloc[:,-1].values.reshape(-1,1))               #Converting the labels vector to a flattened 1 D array  

#Performing the train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)  #Train test split with test size being 30% of original dataset

#Standardizing the features

std_scale = StandardScaler().fit(X_train)              #Fitting the training dataset features
X_train_std = std_scale.transform(X_train)             #Transforming the training dataset features 
X_test_std = std_scale.transform(X_test)               #Transforming the test dataset features

#Perceptron

ppn = Perceptron(max_iter=7, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)      #Creating an instance of the Perceptron Algorithm

ppn.fit(X_train_std, Y_train)                                 # do the training
Y_pred_ppn = ppn.predict(X_test_std).reshape(-1,1)            # Make predictions on the test set
#print('Misclassified samples: %d' % (Y_test != Y_pred_ppn).sum()) # How many samples from the test dataset did the Perceptron algorithm misclassify?
print('Perceptron Test Accuracy: %.2f' % accuracy_score(Y_test, Y_pred_ppn))  # How well did the Perceptron Algorithm do in predicting the test samples?    
print(" ")                      #For better readability on the console

#Combined accuracy
# we perform the stacking operation so we can see how the combination of test and train data did
# NOTE the double parens for hstack and vstack!
X_combined_std = np.vstack((X_train_std, X_test_std))     # vstack puts first array above the second in a vertical stack
y_combined = np.hstack((Y_train, Y_test))                 # hstack puts first array to left of the second in a horizontal stack 
y_combined_pred_ppn = ppn.predict(X_combined_std)         # Making predictions on the combined dataset 
#print('Misclassified combined samples: %d' %  (y_combined != y_combined_pred_ppn).sum())        #How many samples from the combined dataset did the Perceptron Algorithm misclassify
print('Perceptron Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred_ppn))     # Getting Accuracy of the Perceptron Algorithm for the Combined Dataset

print(" ")                     #For better readability on the console

#Logistic Regression

for c_val in [1,10,100]:                            #Running the Logistic Regression ALgorithm for Multiple values of C and Reporting the C value that gives the best accuracy
    lr = LogisticRegression(C=c_val, solver='liblinear', multi_class='ovr', random_state=0)      #Creates an instance of the Logistic Regression Classifier for every value of "C", "ovr" stands for one versus rest
    lr.fit(X_train_std, Y_train) # make the algorithm learn the training data
    # combine the standardized train and test data features
    X_combined_std_lr = np.vstack((X_train_std, X_test_std))    
    Y_combined_lr = np.hstack((Y_train, Y_test))         #Combining the train and test labels
    Y_combined_pred_lr = lr.predict(X_combined_std_lr)    # Performing predictions on the Combined dataset
    acc = accuracy_score(Y_combined_lr, Y_combined_pred_lr)   # Logistic Regression Combined Accuracy   
    print('Logistic Regression Combined Accuracy for C= ', str(c_val),'is: %.2f' % acc)  #Printing Combined Accuracy for each value of C 

print(" ")              #For better readability on the console       

#Support Vector Machine

for c_val in [0.1,1.0,10.0]:                                         #Running the SVM Algorithm for multiple values of C and reporting the C value that gives the best accuracy
    svm = SVC(kernel='linear', C=c_val, random_state=0)              #Creates an instance of svm for each value of "C"
    svm.fit(X_train_std, Y_train)                                 #Make the SVM algorithm learn the training data 
    X_combined_std_svm = np.vstack((X_train_std, X_test_std))     # Combining the train and test dataset features
    Y_combined_svm = np.hstack((Y_train, Y_test))                 # Combining the train and test labels 
    Y_combined_pred_svm = svm.predict(X_combined_std_svm)         # Performing predictions on the combined dataset
    acc_svm = accuracy_score(Y_combined_svm, Y_combined_pred_svm)    # SVM Combined Accuracy 
    print('SVM Combined Accuracy for C= ', str(c_val),'is: %.2f' % acc_svm)    #Printing the Combined Accuracy for each value of C     
    
print(" ")    #For better readability on the console

# Decision  Tree Learning

tree = DecisionTreeClassifier(criterion='entropy',max_depth=5 ,random_state=0)               # Creates an instance of the Decision Tree Algorithm with a max depth of 5 
tree.fit(X_train,Y_train)                                                       #Make the Decision tree learn the test data
X_combined_tree = np.vstack((X_train, X_test))                                  # Combine the train and test features
Y_combined_tree = np.hstack((Y_train, Y_test))                                  # Combine the train and test labels
Y_combined_pred_tree = tree.predict(X_combined_tree)                            # Perform predictions on the combined dataset
acc_tree = accuracy_score(Y_combined_tree, Y_combined_pred_tree)                # Get the combined accuracy
print('Decision Tree Combined Accuracy: %.2f' % acc_tree)                       # Print the combined accuracy to the console
print(" ")                                                                      #For better readability on the console

# Random Forest

for trees in [1, 5, 11]:                                                        # Running Random forest classifier for different values of number of trees/estimators 
    forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, random_state=1, n_jobs=4)   # Creates an instance of the Random Forest Classifier for different values of trees 
    forest.fit(X_train, Y_train)                                               # The algorithm learns the training data
    X_combined_forest = np.vstack((X_train, X_test))                           # Combining the train and test features
    Y_combined_forest = np.hstack((Y_train,Y_test))                            # Combining the train and test labels 
    Y_combined_pred_forest = forest.predict(X_combined_forest)                 # Making predictions on the combined features
    acc_forest = accuracy_score(Y_combined_forest, Y_combined_pred_forest)     # Getting the accuracy for the combined dataset 
    print('Random Forest Combined Accuracy for C= ', str(trees),'is: %.2f' % acc_forest)      #Printing out the combined accuracy for different values of trees

print(" ")                              #For better readability on the console

# K- Nearest Neighbor

for neighs in [1,10,51]:                     # Running the K-NN algorithm for multiple values of "k" neighbors
    knn = KNeighborsClassifier(n_neighbors=neighs, p=2, metric='euclidean')      #Creates an instance of the K-NN algorithm for each value of "k"
    knn.fit(X_train_std, Y_train)                                                # The algorithm learns the training data 
    X_combined_std_knn = np.vstack((X_train_std, X_test_std))                    # Combining the train and test features
    Y_combined_knn = np.hstack((Y_train, Y_test))                                # Combining the train and test labels
    Y_combined_pred_knn = knn.predict(X_combined_std_knn)                        # Making predictions on the combined dataset 
    acc_knn = accuracy_score(Y_combined_knn, Y_combined_pred_knn)                # Getting k-nn algorithm's accuracy for the combined dataset 
    print('K-NN Combined Accuracy for n= ', str(neighs),'is: %.2f' % acc_knn)    # Printing the combined accuracy for each value of "k" 
    