import requests
import pandas as pd
import time
import os

# Seasons we want
seasons = range(2018, 2026)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)


# Function to safely get JSON from the API
def get_json(url):

    for attempt in range(3):

        try:
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "F1-Podium-Predictor"}
            )

            # Check if the server responded successfully
            if response.status_code == 200:
                return response.json()

            print(
                f"Request failed with status "
                f"{response.status_code}. Retrying..."
            )

        except Exception as error:
            print(f"Error: {error}")
            print("Retrying...")

        # Wait before trying again
        time.sleep(3)

    # If all attempts failed
    return None


# Store all seasons here
all_seasons = []


# Go through each season
for season in seasons:

    print(f"\n========== {season} SEASON ==========")

    season_results = []

    # Get the race schedule
    schedule_url = (
        f"https://api.jolpi.ca/ergast/f1/{season}/"
    )

    data = get_json(schedule_url)

    if data is None:
        print(f"Could not get {season} schedule.")
        continue

    races = data["MRData"]["RaceTable"]["Races"]

    # Go through every race
    for race in races:

        round_number = race["round"]
        race_name = race["raceName"]

        print(
            f"Collecting {season} - "
            f"Round {round_number}: {race_name}"
        )

        results_url = (
            f"https://api.jolpi.ca/ergast/f1/"
            f"{season}/{round_number}/results/"
        )

        race_data = get_json(results_url)

        # If the request failed after 3 attempts
        if race_data is None:
            print(
                f"Skipping {season} Round "
                f"{round_number}"
            )
            continue

        races_found = (
            race_data["MRData"]["RaceTable"]["Races"]
        )

        if not races_found:
            print("No results found.")
            continue

        race_results = races_found[0]["Results"]

        # Extract information for every driver
        for driver in race_results:

            season_results.append({

                "season": season,

                "round": int(round_number),

                "race": race_name,

                "position": int(driver["position"]),

                "driver": (
                    driver["Driver"]["givenName"]
                    + " "
                    + driver["Driver"]["familyName"]
                ),

                "constructor": driver["Constructor"]["name"],

                "grid": int(driver["grid"]),

                "points": float(driver["points"]),

                "laps": int(driver["laps"]),

                "status": driver["status"]
            })

        # Wait between API requests
        time.sleep(1)


    # Convert this season to a DataFrame
    season_df = pd.DataFrame(season_results)

    # Save each season separately
    season_file = (
        f"data/f1_race_results_{season}.csv"
    )

    season_df.to_csv(
        season_file,
        index=False
    )

    print(
        f"Saved {season} data: "
        f"{len(season_df)} rows"
    )

    all_seasons.append(season_df)


# Combine all seasons
if all_seasons:

    final_df = pd.concat(
        all_seasons,
        ignore_index=True
    )

    final_file = (
        "data/f1_race_results_2018_2025.csv"
    )

    final_df.to_csv(
        final_file,
        index=False
    )

    print("\n================================")
    print("ALL DATA COLLECTION COMPLETE!")
    print("================================")

    print(
        f"Total rows: {len(final_df)}"
    )

    print(
        f"Total seasons: "
        f"{final_df['season'].nunique()}"
    )

    print(
        f"Total races: "
        f"{final_df[['season', 'round']].drop_duplicates().shape[0]}"
    )

    print(
        f"\nSaved to: {final_file}"
    )