# Detection-of-Counterfeit-Currency

The given data set contains observations based on measurements made on a number of currency bills. The last column is whether the bill is genuine (1) or counterfeit (0). Based on 
the measurements, we build a predictor (Machine Learning Classifier) to determine whether a bill is genuine or counterfeit.
The columns in the database are:

1. Variance of Wavelet Transformed image (continuous) 
2. Skewness of Wavelet Transformed image (continuous) 
3. Kurtosis of Wavelet Transformed image (continuous) 
4. Entropy of image (continuous) 
5. Class label (integer)

The first four columns in the database are the features used to train the classifiers and the last column is just the class label. This project is split into two parts as follows:

## Part 1: Statistical Analysis 

In this phase, we import the dataset using pandas and conduct a statistical study of each variable/feature: correlation of each variable, dependent or independent, with all the other variables. We determine which variables are most highly correlated with each other and also which are highly correlated with the class label. We generate covariance and correlation matrices to show which variables are not independent of each other and which ones are best predictors of genuine money.

The code for this part can be found in the proj1_A.py file. 

## Part 2: Supervised Machine Learning (Classification)

We split the data into training and test datasets and then use the training set to be trained on the following Machine Learning Classifiers:

Perceptron
Logistic Regression
Linear Support Vector Machine
Decision Tree Learning
Random Forest
K-Nearest Neighbors

After the training of each classifier, we make predictions on the test set and get the combined (train + test) accuracy for each classifier. Based on the combined accuracies of all the classifiers, it can be concluded that the Random Forest and the k-Nearest Neighbors classifiers give the best results.

The code for this part can be found in the proj1_B.py file.
