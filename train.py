import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("icu_beds.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").drop_duplicates("date")
print("Date gaps (days):", df["date"].diff().dt.days.value_counts().to_dict())

total = "total_adult_icu_patients"
df["is_weekend"] = (df["date"].dt.day_of_week >= 5).astype(int)
df["change_today"] = df[total] - df[total].shift(1)        
df["change_7days"] = df[total] - df[total].shift(7)        
df["crci_share"] = df["adult_icu_crci_patients"] / df[total]
df["change_tomorrow"] = df[total].shift(-1) - df[total]    
df = df.dropna()

cols = ["change_today", "change_7days", "crci_share", "is_weekend"]
X = df[cols]
y = df["change_tomorrow"]

cut = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]
today_test = df[total].iloc[cut:]
actual_tomorrow = today_test + y_test

model = LinearRegression().fit(X_train, y_train)
pred = today_test + model.predict(X_test)               

print("Baseline (tomorrow = today) MAE:", mean_absolute_error(actual_tomorrow, today_test))
print("Model MAE:", mean_absolute_error(actual_tomorrow, pred))

pickle.dump(model, open("model.pkl", "wb"))