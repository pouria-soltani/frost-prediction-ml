import pandas as pd

df = pd.read_csv("data/records.csv", skiprows=1, parse_dates=["datetime"])
df = df.sort_values("datetime")

required_cols = ["datetime", "tmin", "tmax", "td_m", "um", "ffm"]
df_test = df[required_cols].tail(10).copy()
df_test["datetime"] = df_test["datetime"].dt.strftime("%Y-%m-%d")

df_test.to_csv("test_history.csv", index=False)
print("Saved test_history.csv with", len(df_test), "rows")
print(df_test)
