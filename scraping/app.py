import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time

def fetch_product_image(product_url, save_folder="images", image_number=0):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(product_url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"Failed to fetch page: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, "lxml")

        img_tag = soup.find('img', {'class': '_396cs4 _2amPTt _3qGmMb'})

        if not img_tag:
            print("Image tag not found!")
            return None

        img_url = img_tag.get('src') or img_tag.get('data-src')

        if not img_url:
            print("Image URL not found!")
            return None

        # Download the image
        img_data = requests.get(img_url, headers=headers).content

        # Save the image
        os.makedirs(save_folder, exist_ok=True)
        image_path = os.path.join(save_folder, f"product_{image_number}.jpg")

        with open(image_path, 'wb') as handler:
            handler.write(img_data)

        print(f"Image saved: {image_path}")
        return image_path

    except Exception as e:
        print(f"Error fetching {product_url}: {e}")
        return None

# Load your dataset
dataset_path = "your_dataset.csv"   # <-- update your file name
df = pd.read_csv(dataset_path)

# Assuming your Product Link column is named 'Product Link'
links = df['Product Link'].dropna().tolist()

# Create a folder to save images
save_folder = "downloaded_images"
os.makedirs(save_folder, exist_ok=True)

# New column to store image paths
image_paths = []

# Loop through all product links and fetch images
for idx, link in enumerate(links):
    print(f"Scraping product {idx+1}/{len(links)}")
    img_path = fetch_product_image(link, save_folder, image_number=idx)
    image_paths.append(img_path)
    time.sleep(2)  # Sleep to avoid being blocked

# Add the new column to DataFrame
df['Image Path'] = image_paths

# Save the updated dataset
updated_dataset_path = "your_dataset_with_images.csv"
df.to_csv(updated_dataset_path, index=False)

print(f"\nDataset updated and saved as '{updated_dataset_path}' ✅")
