from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
import time
import re

sys.stdout.reconfigure(encoding='utf-8')

# Define headers for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://www.flipkart.com/search?q=bluetooth+headphones&page={}"
products = []

# Loop through pages 1 and 2
for page in range(1, 3):
    print(f"\nScraping page {page}...")
    try:
        response = requests.get(BASE_URL.format(page), headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Page {page} not found or blocked! Status Code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        items = soup.find_all("div", class_="slAVV4")

        if not items:
            print(f"No items found on page {page}. The structure may have changed or the page may be blocked.")
            break

        # Loop through each product item
        for item in items:
            name_tag = item.find("a", class_="wjcEIp")
            price_tag = item.find("div", class_="Nx9bqj")
            orig_price_tag = item.find("div", class_="yRaY8j")
            rating_tag = item.find("div", class_="XQDdHH")
            num_ratings_tag = item.find("span", class_="Wphh3N")
            link_tag = name_tag['href'] if name_tag and name_tag.has_attr('href') else ""

            if name_tag and price_tag:
                if rating_tag:
                    for img in rating_tag.find_all("img"):
                        img.decompose()

                name = name_tag.get_text(strip=True)
                brand = name.split()[0]
                price = price_tag.get_text(strip=True).replace('₹', '').replace(',', '')
                orig_price = orig_price_tag.get_text(strip=True).replace('₹', '').replace(',', '') if orig_price_tag else price

                try:
                    discount = round(((int(orig_price) - int(price)) / int(orig_price)) * 100)
                except:
                    discount = 0

                rating = rating_tag.get_text(strip=True) if rating_tag else "No Rating"
                num_ratings = num_ratings_tag.get_text(strip=True) if num_ratings_tag else "N/A"
                full_link = "https://www.flipkart.com" + link_tag if link_tag else "N/A"

                # --- Fetch product reviews --- #
                reviews = []
                try:
                    if full_link != "N/A":
                        prod_res = requests.get(full_link, headers=HEADERS, timeout=10)
                        if prod_res.status_code == 200:
                            prod_soup = BeautifulSoup(prod_res.content, "html.parser")
                            # Get reviews section
                            review_tags = prod_soup.find_all("div", class_="RcXBOT")
                            for review_tag in review_tags:
                                review_text = review_tag.find("div", class_="z9E0IG").get_text(strip=True) if review_tag.find("div", class_="z9E0IG") else "No Review"
                                reviews.append(review_text)

                        time.sleep(1.5)
                except Exception as e:
                    print(f"Error fetching reviews for {name}: {e}")

                # Append product details to the list
                products.append({
                    "Product Name": name,
                    "Brand": brand,
                    "Price": int(price),
                    "Original Price": int(orig_price),
                    "Discount %": discount,
                    "Rating": rating,
                    "Number of Ratings": num_ratings,
                    "Product Link": full_link,
                    "Reviews": "; ".join(reviews)  # Join all reviews into a single string
                })

        time.sleep(2)

    except Exception as e:
        print(f"Error scraping page {page}: {e}")
        continue

# Save to CSV
df = pd.DataFrame(products)
print(f"\nTotal products scraped: {len(df)}")
df.to_csv("flipkart_headphones_with_reviews.csv", mode='w', header=True, index=False)
