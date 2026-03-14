import pandas as pd
import numpy as np

def clip(df, col="RevolvingUtilizationOfUnsecuredLines", bounds=(0,1)) -> pd.DataFrame:
  df = df.copy()
  low, high = bounds
  series = df[col]
  count = ((series < low) | (series > high)).sum()
  df[col] = series.clip(low, high)
  print(f"{count} values clipped in '{col}'")
  return df

def impute(df, feature, medians_from_train_df=None) -> pd.DataFrame:
  df = df.copy()
  age_bins = [0, 30, 45, 60, 75, 120]
  age_groups = pd.cut(df['age'], bins=age_bins)
  if medians_from_train_df is not None:
    df[feature] = df[feature].fillna(age_groups.map(medians_from_train_df))
  else:
    df[feature] = df.groupby(age_groups)[feature].transform(lambda x: x.fillna(x.median()))
  return df

def get_impute_stats(df, feature) -> pd.Series:
  age_bins = [0, 30, 45, 60, 75, 120]
  return df.groupby(pd.cut(df['age'], bins=age_bins))[feature].median()

def feature_engieneering(df) -> pd.DataFrame:
  df = df.copy()
  df['MonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
  df['TotalLatePayments'] = df['NumberOfTime30-59DaysPastDueNotWorse'] \
  + df['NumberOfTime60-89DaysPastDueNotWorse'] + df['NumberOfTimes90DaysLate']

  return df

def remove_first_column(df) -> pd.DataFrame:
  df = df.copy()
  df.drop('Unnamed: 0', axis=1, inplace=True)
  # axis=1 must be added to remove column not a row (axis=0)
  return df

def remove_target_column(df) -> pd.Dataframe:
  df = df.copy()
  df.drop('SeriousDlqin2yrs', axis=1, inplace=True)
  return df

def save_processed(df, path) -> None:
  df.to_csv(path, index=False)
  print(f"Saved {len(df)} rows to {path}")