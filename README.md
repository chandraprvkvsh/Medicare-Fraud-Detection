# Analysis and Modeling of Medicare Fraud Involving Healthcare Providers, Physicians, and Beneficiaries

Medicare fraud poses a significant challenge, driving up healthcare costs and impacting insurance premiums. This project aims to tackle this issue by leveraging predictive analytics to identify potentially fraudulent providers based on claims data. By analyzing patterns in claims submissions and beneficiary details, the project seeks to uncover key variables indicative of fraudulent behavior. Through rigorous analysis of inpatient and outpatient claims, along with beneficiary information, the goal is to develop insights that can help detect and prevent fraudulent activities, ultimately safeguarding the integrity of the healthcare system.

The dataset consists of 8 .csv files out of which 4 correspond to the training dataset and 4 correspond to the test dataset:

| Dataset            | Description                                           |
|--------------------|-------------------------------------------------------|
| Train_Inpatient    | Claims made by people who were hospitalized          |
| Train_Outpatient   | Claims made by people who were not hospitalized      |
| Train_Beneficiary  | Beneficiary details                                   |
| Train_Labels       | Ground truth label of "PotentialFraud" for Providers |

The not-common columns between Train_Inpatient and Train_Outpatient are:

- AdmissionDt (Train_Inpatient only)
- DischargeDt (Train_Inpatient only)
- DiagnosisGroupCode (Train_Inpatient only)

Apart from these, all columns are common in both datasets. Therefore, we can add a binary feature named Hospitalization to both dataframes and merge them.

To simplify the process further, we can join the other two dataframes with the already merged one to consolidate all information.

**Note:** The test dataset follows the same structure as the training dataset, enabling us to apply all transformations and changes in the training set to the test set by re-using the same code.

Before starting the Exploratory Data Analysis (EDA), it's beneficial to make some basic changes to the dataframe:
1. Remove DOD (Date of Death) and DOB (Date of Birth), and add Age and Alive as new features.
2. Remove ClaimStart and ClaimEnd dates, and add the time period.
3. Remove AdmitDate and DischargeDate, and add the time period.
4. Change values of ChronicConditions variables from '2' to '0' to make it binary (0 or 1).
5. Do the same for Gender.
6. Replace PotentiallyFraud with 1 and not with 0.
7. Replace RenalDiseaseIndicator values accordingly.

Observing that there are three types of Physicians: Attending, Operating, and Other, we identify instances where a single physician serves more than one role. We add two extra features:
- Count_Physician: Total number of physicians working.
- Check_Physician: A discrete feature {0,1,2,3} capturing uniquely if there are common physicians serving multiple roles.

**Note:** There is a significant repetition of Providers in the entire dataset. To address this, we need to accumulate meaningful features and have a unique set of feature values for each provider.

After visualization and checking the unique values of features, we:
- Accumulate features to have a single set for each unique Provider.
- Address class imbalance.
- Perform Z-Score Normalization.
- Try a couple of classifiers while hyperparameter tuning XGBoost and RandomForest.
- Attempt a Feedforward Neural Network, but validation loss fluctuates, possibly due to overfitting or model inability to generalize well.
- Save the XGBoost model and use it to make predictions on the test set.
