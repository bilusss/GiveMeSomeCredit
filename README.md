# GiveMeSomeCredit

## My goal

TLDR: I want to predict if my client will have trouble paying off their debt based on historical client data.



## Data description


* ~250k examples

* 150k training (~10k positive, ~140k negative)


### important mentions about this exact dataset:

#### **UNBALANCED DATA** 

only 7.16% positive, meaning that accuracy will not work out in this scenerio (model would predict negative and still achive +90% accuracy) so AUC (Area Under Curve) is a better metric

#### **NULLs** 

MonthlyIncome has null in 19.82% of examples, NumberOfDependents has 2.62%, i'll use median in both cases



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

---

## My approach

Since in this dataset I got labeled data supervised learning classification models are a natural initial approach.

* [✅] First step is to handle missing values and explore the data
* [✅] asdasd


---

### Tools used

* python
* pandas
* sklearn
* numpy

