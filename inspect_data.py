import pandas as pd 
df=pd.read_csv('icu_beds.csv')
print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())
