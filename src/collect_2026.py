import requests
import pandas as pd
import time
import os

# ============================================================
# COLLECT 2026 F1 RACE RESULTS BEFORE SINGAPORE
# ============================================================

BASE_URL = "https://api.jolpi.ca/ergast/f1"

OUTPUT_FILE = "data/f1_race_results_2026.csv"

all_results = []

print("=" * 60)
print("COLLECTING 2026 F1 RACE RESULTS")
print("=" * 60)

# Singapore is Round 17.
# We only want completed races: Rounds 1-16.
for round_no in range(1, 17):

    url = f"{BASE_URL}/2026/{round_no}/results.json"

    print(f"\nFetching Round {round_no}...")

    try:
        response = requests.get(url, timeout=20)

        if response.status_code != 200:
            print(f"  Failed: HTTP {response.status_code}")
            continue

        data = response.json()

        races = data["MRData"]["RaceTable"]["Races"]

        if not races:
            print("  No race data found.")
            continue

        race = races[0]

        race_name = race["raceName"]
        circuit_name = race["Circuit"]["circuitName"]
        date = race["date"]

        results = race["Results"]

        print(f"  Race: {race_name}")
        print(f"  Date: {date}")
        print(f"  Drivers: {len(results)}")

        for result in results:

            driver = result["Driver"]

            all_results.append({
                "season": 2026,
                "round": int(round_no),
                "race": race_name,
                "date": date,
                "driver": (
                    driver["givenName"]
                    + " "
                    + driver["familyName"]
                ),
                "driver_id": driver["driverId"],
                "constructor": result["Constructor"]["name"],
                "constructor_id": result["Constructor"]["constructorId"],
                "grid": int(result["grid"]),
                "position": (
                    int(result["position"])
                    if result["position"].isdigit()
                    else None
                ),
                "points": float(result["points"]),
                "laps": int(result["laps"]),
                "status": result["status"],
                "fastest_lap": (
                    float(result["FastestLap"]["AverageSpeed"]["speed"])
                    if "FastestLap" in result
                    and "AverageSpeed" in result["FastestLap"]
                    else None
                ),
                "circuit": circuit_name
            })

        time.sleep(0.5)

    except Exception as e:
        print(f"  ERROR: {e}")


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_results)

if df.empty:
    print("\nERROR: No data was collected.")
    print("Check your internet connection or API URL.")
    raise SystemExit

# Remove accidental duplicates
df = df.drop_duplicates(
    subset=["season", "round", "driver_id"]
)

# Sort properly
df = df.sort_values(
    ["round", "position"],
    na_position="last"
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

os.makedirs("data", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("2026 DATA COLLECTION COMPLETE")
print("=" * 60)

print(f"\nRows collected: {len(df)}")
print(f"Races collected: {df['round'].nunique()}")

print("\nRaces:")
print(
    df[
        ["round", "race", "date"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nTop 10 rows:")
print(df.head(10).to_string(index=False))

print(f"\nSaved to:")
print(OUTPUT_FILE)