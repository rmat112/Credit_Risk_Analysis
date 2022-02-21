# Credit_Risk_Analysis
## Overview of the Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to employ different 
techniques to train and evaluate models with unbalanced classes. We will use imbalanced-learn and scikit-learn libraries to build and evaluate 
models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler 
and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling 
using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, 
to predict credit risk. Then we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Resources
- Data Source: Lending Club [LoanStats_2019Q1.csv](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)<br/>
- Software: Python 3.9.7, Jupyter Notebooks 6.4.5

## Results
## 1. Using Resampling Models to Predict Credit Risk 
Jupyter Notebook: [credit_risk_resampling_code.ipynb](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/credit_risk_resampling_code.ipynb)<br/>
Balanced accuracy score, confusion matrix, and classification report are generated as shown below:<br/>
### 1a. RandomOverSampler Algorithm
![ROS(1).png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/ROS(1).png)

As seen above, for high risk case:<br/>
- Balanced accuracy score is calculated at 64.6%
- Precision score is 1%, 
- Sensitivity is 61%, and 
- F1 score is 2%

### 1b. SMOTE Algorithm
![Smote.png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/Smote.png)

As seen above, for high risk case:<br/>
- Balanced accuracy score is calculated at 62.3%
- Precision score is 1%, 
- Sensitivity is 61%, and 
- F1 score is 2%

### 1c. Under sampling ClustCentroids Algorithm
![CC.png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/CC.png)

As seen above, for high risk case:<br/>
- Balanced accuracy score is calculated at 51.3%
- Precision score is 1%, 
- Sensitivity is 60%, and 
- F1 score is 1%

## 2. Using the SMOTEENN algorithm to Predict Credit Risk
Jupyter Notebook: [credit_risk_resampling_code.ipynb](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/credit_risk_resampling_code.ipynb)<br/>
Balanced accuracy score, confusion matrix, and classification report are generated as shown below:<br/>
![Smoteen.png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/Smoteen.png)

As seen above, for high risk case:<br/>
- Balanced accuracy score is calculated at 61.6%
- Precision score is 1%, 
- Sensitivity is 69%, and 
- F1 score is 2%


## 3. Using Ensemble Classifiers to Predict Credit Risk
Jupyter Notebook: [credit_risk_ensemble.ipynb](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)<br/>
Balanced accuracy score, confusion matrix, and classification report are generated as shown below:<br/>
### 3a. BalancedRandomForestClassifier Algorithm
![BRFC.png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/BRFC.png)

As seen above, for high risk case:<br/>
- Balanced accuracy score is calculated at 78.8%
- Precision score is 4%, 
- Sensitivity is 67%, and 
- F1 score is 7%

### The features are sorted in descending order by feature importance
![BRFC(2).png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/BRFC(2).png)


### 3b. EasyEnsembleClassifier Algorithm
![EEC.png](https://github.com/rmat112/Credit_Risk_Analysis/blob/main/Images/EEC.png)

As seen above, for high risk case:<br/>
- Balanced accuracy score is calculated at 92.5%
- Precision score is 7%, 
- Sensitivity is 91%, and 
- F1 score is 14%


# Summary
Presented below is a summary of the balanced accuracy scores, the precision, sensitivity, and the F1 scores for all six models in a descending order(based on high risk applications):
- EasyEnsembleClassifier:         Accuracy 92.5%, Precision 7%, Sensitivity 91%, F1 score 14%
- BalancedRandomForestClassifier: Accuracy 78.8%, Precision 4%, Sensitivity 67%, F1 score 7%
- RandomOverSampler:              Accuracy 64.6%, Precision 1%, Sensitivity 61%, F1 score 2%
- SMOTE:                          Accuracy 62.3%, Precision 1%, Sensitivity 61%, F1 score 2%
- SMOTEEN:                        Accuracy 61.6%, Precision 1%, Sensitivity 69%, F1 score 2%
- ClusterCentroids:               Accuracy 51.3%, Precision 1%, Sensitivity 60%, F1 score 1%

## Recommendation
It appears that the EasyEnsembleClassifier should be the model of choice as it has an accuracy score of 92.5%. The sensitivity is also very high for this model, which means that rate at which high risk applications are correctly predicted is high. However, I would not recommend any of these models because the precision rate is still very low (7% or below), which means that rate of high risk predictions being correct is still very low. This would result in unnecessary loss of business. Therefore, we should look for another better model.
