import logging
import os
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    # Chrome on Windows
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    # Safari on Mac
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    # Chrome on Linux
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
    # Safari on iPhone
]

base_url = "https://www.zameen.com/Flats_Apartments/Karachi-2-{}.html"

complete_property_data = []

for page_num in range(1, 401):
    url = base_url.format(page_num)
    headers = {
        "User-Agent": random.choice(user_agents)
    }

    logger.info("Scraping Page %d", page_num)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch page %d: %s", page_num, e)
        continue

    soup = BeautifulSoup(response.content, "html.parser")

    listings = soup.find_all("li", {"role": "article"})

    for listing in listings:
        try:
            title = listing.find("h2", {"aria-label": "Title"})
            title = title.text.strip() if title else None

            price = listing.find("span", {"aria-label": "Price"})
            price = price.text.strip() if price else None

            location = listing.find("div", {"aria-label": "Location"})
            location = location.text.strip() if location else None

            beds = listing.find("span", {"aria-label": "Beds"})
            beds = beds.text.strip() if beds else None

            baths = listing.find("span", {"aria-label": "Baths"})
            baths = baths.text.strip() if baths else None

            area = listing.find("span", {"aria-label": "Area"})
            area = area.text.strip() if area else None

            try:
                date_added = listing.find("span", {"aria-label": "Listing creation date"}).text.strip()
            except AttributeError:
                date_added = ""

            data = {
                "Type": "Apartment",
                "Title": title,
                "Price": price,
                "Location": location,
                "Beds": beds,
                "Baths": baths,
                "Area": area,
                "Date": date_added
            }


            complete_property_data.append(data)

        except AttributeError as e:
            logger.warning("Error parsing a listing: %s", e)


    delay = random.uniform(2, 3)
    time.sleep(delay)


df = pd.DataFrame(complete_property_data)

file = os.path.join(os.path.dirname(__file__), "data", "property-data.csv")
os.makedirs(os.path.dirname(file), exist_ok=True)

if os.path.exists(file):
    df.to_csv(file, mode='a', index=False, header=False)
else:
    df.to_csv(file, index=False)

logger.info("Scraping completed! Saved %d properties to %s", len(complete_property_data), file)