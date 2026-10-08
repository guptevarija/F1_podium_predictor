import pandas as pd

# Load race results
race_file = "data/f1_race_results_2018_2025.csv"

# Load qualifying results
qualifying_file = "data/f1_qualifying_results_2018_2025.csv"

race_df = pd.read_csv(race_file)
qualifying_df = pd.read_csv(qualifying_file)

print("Race data rows:", len(race_df))
print("Qualifying data rows:", len(qualifying_df))


# Columns that uniquely identify a driver in a particular race
merge_columns = [
    "season",
    "round",
    "driver"
]


# Merge race and qualifying data
df = pd.merge(
    race_df,
    qualifying_df,
    on=merge_columns,
    how="left",
    suffixes=("_race", "_qualifying")
)


# Create our target variable
# 1 = podium
# 0 = no podium

df["podium"] = (
    df["position"] <= 3
).astype(int)


# Save the combined dataset

output_file = (
    "data/f1_master_dataset_2018_2025.csv"
)

df.to_csv(
    output_file,
    index=False
)


# Display information

print("\n================================")
print("MASTER DATASET CREATED!")
print("================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nPodium distribution:")
print(df["podium"].value_counts())

print("\nMissing qualifying positions:")
print(
    df["qualifying_position"].isna().sum()
)

print("\nSaved to:")
print(output_file)