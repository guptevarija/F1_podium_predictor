import requests
import pandas as pd
import time
import os

# Seasons to collect
seasons = range(2018, 2026)

# Make sure data folder exists
os.makedirs("data", exist_ok=True)


def get_json(url):

    for attempt in range(3):

        try:
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "F1-Podium-Predictor"}
            )

            if response.status_code == 200:
                return response.json()

            print(
                f"Request failed: {response.status_code}. "
                f"Retrying..."
            )

        except Exception as error:
            print(f"Error: {error}")

        time.sleep(3)

    return None


all_results = []


for season in seasons:

    print(f"\n========== {season} SEASON ==========")

    schedule_url = (
        f"https://api.jolpi.ca/ergast/f1/{season}/"
    )

    data = get_json(schedule_url)

    if data is None:
        print(f"Could not get {season} schedule.")
        continue

    races = data["MRData"]["RaceTable"]["Races"]


    for race in races:

        round_number = race["round"]
        race_name = race["raceName"]

        print(
            f"Collecting {season} - "
            f"Round {round_number}: {race_name}"
        )

        qualifying_url = (
            f"https://api.jolpi.ca/ergast/f1/"
            f"{season}/{round_number}/qualifying/"
        )

        qualifying_data = get_json(qualifying_url)

        if qualifying_data is None:
            print("Could not collect qualifying.")
            continue

        races_found = (
            qualifying_data["MRData"]
            ["RaceTable"]
            ["Races"]
        )

        if not races_found:
            print("No qualifying data found.")
            continue

        qualifying_results = races_found[0]["QualifyingResults"]


        for driver in qualifying_results:

            q1 = driver.get("Q1")
            q2 = driver.get("Q2")
            q3 = driver.get("Q3")

            all_results.append({

                "season": season,

                "round": int(round_number),

                "race": race_name,

                "qualifying_position": int(
                    driver["position"]
                ),

                "driver": (
                    driver["Driver"]["givenName"]
                    + " "
                    + driver["Driver"]["familyName"]
                ),

                "constructor": driver["Constructor"]["name"],

                "Q1": q1,

                "Q2": q2,

                "Q3": q3
            })


        time.sleep(1)


# Convert to DataFrame
df = pd.DataFrame(all_results)


# Save
file_path = (
    "data/f1_qualifying_results_2018_2025.csv"
)

df.to_csv(
    file_path,
    index=False
)


print("\n================================")
print("QUALIFYING DATA COMPLETE!")
print("================================")

print(f"Total rows: {len(df)}")

print(
    f"Total seasons: "
    f"{df['season'].nunique()}"
)

print(
    f"Total races: "
    f"{df[['season', 'round']].drop_duplicates().shape[0]}"
)

print(f"Saved to: {file_path}")