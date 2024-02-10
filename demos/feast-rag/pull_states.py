from typing import Dict, List
import wikipedia as wiki
import pandas as pd

CITIES = [
    "New York, New York",
    "Los Angeles, California",
    "Chicago, Illinois",
    "Houston, Texas",
    "Phoenix, Arizona",
    "Philadelphia, Pennsylvania",
    "San Antonio, Texas",
    "San Diego, California",
    "Dallas, Texas",
    "San Jose, California",
    "Austin, Texas",
    "Jacksonville, Florida",
    "Fort Worth, Texas",
    "Columbus, Ohio",
    "Charlotte, North Carolina",
    "San Francisco, California",
    "Indianapolis, Indiana",
    "Seattle, Washington",
    "Denver, Colorado",
    "Washington, D.C.",
    "Boston, Massachusetts",
    "El Paso, Texas",
    "Nashville, Tennessee",
    "Detroit, Michigan",
    "Oklahoma City, Oklahoma",
    "Portland, Oregon",
    "Las Vegas, Nevada",
    "Memphis, Tennessee",
    "Louisville, Kentucky",
    "Baltimore, Maryland",
    "Milwaukee, Wisconsin",
    "Albuquerque, New Mexico",
    "Tucson, Arizona",
    "Fresno, California",
    "Mesa, Arizona",
    "Sacramento, California",
    "Atlanta, Georgia",
    "Kansas City, Missouri",
    "Colorado Springs, Colorado",
    "Miami, Florida",
    "Raleigh, North Carolina",
    "Omaha, Nebraska",
    "Long Beach, California",
    "Virginia Beach, Virginia",
    "Oakland, California",
    "Minneapolis, Minnesota",
    "Tulsa, Oklahoma",
    "Arlington, Texas",
    "Tampa, Florida",
    "New Orleans, Louisiana"
]

def get_wikipedia_summary(cities: List[str]) -> Dict[str, str]:
    city_summary_dict = {}
    for city in cities:
        try:
            city_summary_dict[city] = wiki.summary(city)
        except:
            print(f"error retrieving {city}")

    return city_summary_dict


def write_data(output_dict: Dict[str, str]) -> None:
    df = pd.DataFrame([output_dict]).T.reset_index()
    df.columns = ['State', 'Wiki Summary']
    df.to_csv("city_wikipedia_summaries.csv", index=False)

def main():
    city_dict_output = get_wikipedia_summary(CITIES)
    write_data(city_dict_output)

if __name__ == "__main__":
    main()
