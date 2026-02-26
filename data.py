import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("--- Dataset statistics ---")
print(df.describe())

print("\n--- Feature Correlation Matrix ---")
print(df.drop('species', axis=1).corr())

run_stats = True
run_corr = True

if run_stats:
    print("\n[Branch] Dataset statistics executed")

if run_corr:
    print("\n[Branch] Feature correlation executed")