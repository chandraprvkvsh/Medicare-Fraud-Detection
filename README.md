# FraudDetection

The dataset consists of 8 .csv files out of which 4 correspond to training dataset and 4 correspond to test dataset

Train_Inpatient   | Consists of claims made by people who were hospitalized 
Train_Outpatient  | Consists of claims made by people who were not hospitalized
Train_Beneficiary | Consists of Beneficiary details
Train_Labels      | Consists of the ground truth label of "PotentialFraud" for all Providers that are there in the dataset



The not-common columns between Train_Inpatient, Train_Outpatient are:

AdmissionDt (Train_Inpatient only)
DischargeDt (Train_Inpatient only)
DiagnosisGroupCode (Train_Inpatient only)

Apart from these, All columns are common in both so we can add a binary feature named Hospitalization to both dataframes and merge them.



To make the further process simpler for us, We can just join the other two dataframes also with the already merged one to get all information at one place.

NOTE : We observed that the test dataset is following the very same structure so we can easily apply all transformations, changes in train set to test set by re-using the same code.



Before starting with the EDA part, It'll be beneficial for us to just do some basic changes to the dataframe:
1) Remove DOD, DOB and just add Age, Alive as the new features.
2) Remove ClaimStart, ClaimEnd date, Add the time period.
3) Remove AdmitDate, DischargedDate, Add the time period.
4) It is found that ChronicConditions variables has '2', We can change those to '0' in order to make it {0,1}
5) The same thing can be done with Gender as well
6) We can replace PotentiallyFraud with 1 and not with 0
7) Similar thing for RenalDiseaseIndicator



As we observe now that there are three types of Physicians: Attending/ Operating/ Other

In some cases a single physicist is serving more than one of these roles, So we analyse such occurences and add two extra features:

Count_Physician : It counts the total number of physicians working
Check_Physician : It is a discrete feature {0,1,2,3} which captures uniquely if there are common physicians serving multiple roles



*NOTE : There is a huge repetition of Providers in the entire dataset (And not just because of the outer join, The difference would be still there if we considered any original dataframe and compared the number of rows to the total number of providers), So we'll need to accumulate meaningful features and somehow have a unique set of feature values for each provider.



After some visualization and check of how many unique values some of these features has, We accumulated features in order to have single set of features for each unique Provider, Addressed class imbalance, Did Z-Score Normalization and tried a couple of classifiers while hyperparameter tuning XGBoost and RandomForest. Also tried Feedforward Neural Network, But the validation loss fluctuated probably because of some potential overfitting or disability of model to generalise well. So, Saved XGBoost model and used that to make predictions on the test set.

