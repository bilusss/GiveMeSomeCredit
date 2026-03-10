import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# axis=1 means columns, 0 rows

train_df = train_df.fillna(train_df.median(numeric_only=True))
# Can be done also this way \/
# train_df['MonthlyIncome'].fillna(train_df['MonthlyIncome'].median(), inplace=True)



# Cleaning test data (same way) preventing data leakage - using only training set median: 

test_df = test_df.drop('Unnamed: 0', axis=1)
test_df = test_df.fillna(train_df.median(numeric_only=True))

# Verifing shapes match

# print(f"Train shape: {train_df.shape}")
# print(f"Test shape: {test_df.shape}")

# print(train_df.info())
# train_df.head()
# columns = train_df.columns.to_list()
# print(columns, end='\n\n')
# for column in columns:
#   print(train_df[column].describe(), end='\n\n')
# train_df[['MonthlyIncome', 'DebtRatio', 'age']].describe()

# Getting rid off age = 0:
train_df = train_df[train_df['age'] >= 18]
test_df = test_df[test_df['age']>=18]


"""
X_train = train_df.drop('SeriousDlqin2yrs', axis=1)
y_train = train_df['SeriousDlqin2yrs']

X_test = test_df.drop('SeriousDlqin2yrs', axis=1)
y_test = test_df['SeriousDlqin2yrs']

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # ważne! 7% positive rate
        random_state=42,
        n_jobs=-1  # używa wszystkich CPU
    ))
])

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score}")

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-val AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Save predictions in sampleEntry.csv format
y_pred_proba = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'Id': range(1, len(y_pred_proba) + 1),
    'Probability': y_pred_proba
})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")


"""