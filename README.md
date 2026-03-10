# GiveMeSomeCredit

## My goal

TLDR: I want to predict if my client will have trouble paying off their debt based on historical client data.

Current **AUC**: ```0.84149```,
0.5 - random model, 0.5-1.0 - better than random, 1.0 - perfect model

Current **GINI**: ```2 * AUC - 1 = 0.68298```, 
currently my model has ~68% advantage over randomness

* I'm still trying to impove the model


## Data description


* ~250k examples

* 150k training (~10k positive, ~140k negative)


### important mentions about this exact dataset:

#### **UNBALANCED DATA** 

only 7.16% positive, meaning that accuracy will not work out in this scenerio (model would predict negative and still achive +90% accuracy) so AUC (Area Under Curve) is a better metric

#### **NULLs** 

MonthlyIncome has null in 19.82% of examples, NumberOfDependents has 2.62%, i'll use median in both cases


### variables description


| Variable Name | Description | Type |
| :--- | --- | ---: |
| **SeriousDlqin2yrs (target)** | **Person experienced 90 days past due delinquency or worse** | **Y/N** |
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits | percentage |
| age | Age of borrower in years | integer |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years. | integer |
| DebtRatio | Monthly debt payments, alimony, living costs divided by monthly gross income | percentage |
| MonthlyIncome | Monthly income | real |
| NumberOfOpenCreditLinesAndLoans | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards) | integer |
| NumberOfTimes90DaysLate | Number of times borrower has been 90 days or more past due. | integer |
| NumberRealEstateLoansOrLines | Number of mortgage and real estate loans including home equity lines of credit | integer |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years. | integer |
| NumberOfDependents | Number of dependents in family excluding themselves (spouse, children etc.) | integer |

> This table above is originally attached to the [dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview)

### Dataset statistics (train_df.describe())

| Statystyka | SeriousDlqin2yrs | RevolvingUtilization | age | NumberOfTime30-59 | DebtRatio | MonthlyIncome | NumberOfOpenCredit | NumberOfTimes90DaysLate | NumberRealEstate | NumberOfTime60-89 | NumberOfDependents |
|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 150000 | 150000 | 150000 | 150000 | 150000 | 120269 | 150000 | 150000 | 150000 | 150000 | 146076 |
| mean | 0.067 | 6.048 | 52.30 | 0.421 | 353.005 | 6670.22 | 8.453 | 0.266 | 1.018 | 0.240 | 0.757 |
| std | 0.250 | 249.755 | 14.77 | 4.193 | 2037.82 | 14384.67 | 5.146 | 4.169 | 1.130 | 4.155 | 1.115 |
| min | - | - | - | - | - | - | - | - | - | - | - |
| 25% | - | 0.030 | 41.0 | - | 0.175 | 3400.0 | 5.0 | - | - | - | - |
| 50% | - | 0.154 | 52.0 | - | 0.367 | 5400.0 | 8.0 | - | 1.0 | - | - |
| 75% | - | 0.559 | 63.0 | - | 0.868 | 8249.0 | 11.0 | - | 2.0 | - | 1.0 |
| max | 1.0 | 50708.0 | 109.0 | 98.0 | 329664.0 | 3008750.0 | 58.0 | 98.0 | 54.0 | 98.0 | 20.0 |

---

## My approach

Since in this dataset I got labeled data supervised learning classification models are a natural initial approach.

* [✅] First step is to handle missing values and explore the data
* [✅] 


---

### Tools used

* python
* pandas
* sklearn
* numpy
* seaborn