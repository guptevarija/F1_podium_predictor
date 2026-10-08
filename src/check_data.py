import pandas as pd

# Load our F1 dataset
file_path = "data/f1_race_results_2018_2025.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\n========== SEASONS ==========")

print(df["season"].value_counts().sort_index())

print("\n========== RACES PER SEASON ==========")

races_per_season = (
    df.groupby("season")["round"]
    .nunique()
)

print(races_per_season)

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("\n========== FIRST 10 ROWS ==========")

print(df.head(10))

print("\n========== LAST 10 ROWS ==========")

print(df.tail(10))