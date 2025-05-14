from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
import time
import re

# Fix encoding for ₹ and special characters
sys.stdout.reconfigure(encoding='utf-8')

# Proper headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://www.flipkart.com/search?q=bluetooth+headphones&page={}"
products = []

for page in range(1,30):  # Continue scraping pages
    print(f"\nScraping page {page}...")
    try:
        response = requests.get(BASE_URL.format(page), headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print("Page not found or blocked!")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        items = soup.find_all("div", class_="slAVV4")

        if not items:
            print("No items found. Possibly structure changed or blocked.")
            break

        for item in items:
            name_tag = item.find("a", class_="wjcEIp")
            price_tag = item.find("div", class_="Nx9bqj")
            orig_price_tag = item.find("div", class_="yRaY8j")  # strike-through price
            rating_tag = item.find("div", class_="XQDdHH")
            num_ratings_tag = item.find("span", class_="Wphh3N")  # number of ratings
            link_tag = name_tag['href'] if name_tag and name_tag.has_attr('href') else ""

            if name_tag and price_tag:
                # Clean rating if images inside
                if rating_tag:
                    for img in rating_tag.find_all("img"):
                        img.decompose()

                name = name_tag.get_text(strip=True)
                brand = name.split()[0]  # Assuming first word is brand (common practice)

                # Prices
                price = price_tag.get_text(strip=True).replace('₹', '').replace(',', '')
                orig_price = orig_price_tag.get_text(strip=True).replace('₹', '').replace(',', '') if orig_price_tag else price

                # Discount calculation
                try:
                    discount = round(((int(orig_price) - int(price)) / int(orig_price)) * 100)
                except:
                    discount = 0

                # Ratings
                rating = rating_tag.get_text(strip=True) if rating_tag else "No Rating"
                num_ratings = num_ratings_tag.get_text(strip=True) if num_ratings_tag else "N/A"

                full_link = "https://www.flipkart.com" + link_tag if link_tag else "N/A"

                products.append({
                    "Product Name": name,
                    "Brand": brand,
                    "Price": int(price),
                    "Original Price": int(orig_price),
                    "Discount %": discount,
                    "Rating": rating,
                    "Number of Ratings": num_ratings,
                    "Product Link": full_link
                })

        time.sleep(1.5)

    except Exception as e:
        print("Error scraping page:", page, str(e))
        continue

# Save to CSV
df = pd.DataFrame(products)
print(f"\nTotal products scraped: {len(df)}")
df.to_csv("flipkart_headphones.csv", mode='a', header=True, index=False)
