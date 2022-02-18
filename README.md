# Credit_Risk_Analysis
# Overview of the Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to employ different 
techniques to train and evaluate models with unbalanced classes. We will use imbalanced-learn and scikit-learn libraries to build and evaluate 
models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler 
and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling 
using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, 
to predict credit risk. Then we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

# Results
## 1. Using Resampling Models to Predict Credit Risk 
#### An accuracy score for the model is calculated
#### A confusion matrix has been generated
#### An imbalanced classification report has been generated


## 2. Using the SMOTEENN algorithm to Predict Credit Risk
#### An accuracy score for the model is calculated
#### A confusion matrix has been generated
#### An imbalanced classification report has been generated


## 3. Using Ensemble Classifiers to Predict Credit Risk
### 3a. BalancedRandomForestClassifier
#### An accuracy score for the model is calculated
#### A confusion matrix has been generated
#### An imbalanced classification report has been generated
#### The features are sorted in descending order by feature importance

### 3b. EasyEnsembleClassifier
#### An accuracy score for the model is calculated
#### A confusion matrix has been generated
#### An imbalanced classification report has been generated


# Summary
## Recommendation or Justification
