import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('cs-training.csv',delimiter=',')
test_df = pd.read_csv('cs-test.csv',delimiter=',')

# print(train_df.info())
# print(train_df.head())


# Let's check if there is any missing data:

# missing_percent_train = (train_df.isnull().sum() / len(train_df)) * 100
# print(missing_percent_train)

# missing_percent_test = (train_df.isnull().sum() / len(train_df)) * 100
# print(missing_percent_test)


# Cleaning training data: 

train_df = train_df.drop('Unnamed: 0', axis=1)
# A list can be passed instead of a string

train_df = train_df.fillna(train_df.median(numeric_only=True))
# Can be done also this way \/
# train_df['MonthlyIncome'].fillna(train_df['MonthlyIncome'].median(), inplace=True)



# Cleaning test data (same way) preventing data leakage - using only training set median: 

test_df = test_df.drop('Unnamed: 0', axis=1)
test_df = test_df.fillna(train_df.median(numeric_only=True))

# Verifing shapes match

# print(f"Train shape: {train_df.shape}")
# print(f"Test shape: {test_df.shape}")

X_train = train_df.drop('SeriousDlqin2yrs', axis=1)
y_train = train_df['SeriousDlqin2yrs']

X_test = test_df.drop('SeriousDlqin2yrs', axis=1)
y_test = test_df['SeriousDlqin2yrs']



