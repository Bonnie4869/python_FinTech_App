import json
import csv
import requests
import time
from typing import List, Dict, Optional

# Coordinates of Hong Kong's 18 districts
DISTRICT_COORDS = [
    {"name": "Central_and_Western", "lat": "22.2819", "lng": "114.1586"},
    {"name": "Wan_Chai", "lat": "22.2783", "lng": "114.1733"},
    {"name": "Eastern", "lat": "22.2849", "lng": "114.2249"},
    {"name": "Southern", "lat": "22.2394", "lng": "114.1576"},
    {"name": "Yau_Tsim_Mong", "lat": "22.3040", "lng": "114.1710"},
    {"name": "Sham_Shui_Po", "lat": "22.3316", "lng": "114.1624"},
    {"name": "Kowloon_City", "lat": "22.3282", "lng": "114.1913"},
    {"name": "Wong_Tai_Sin", "lat": "22.3422", "lng": "114.2036"},
    {"name": "Kwun_Tong", "lat": "22.3134", "lng": "114.2251"},
    {"name": "Tsuen_Wan", "lat": "22.3705", "lng": "114.1048"},
    {"name": "Tuen_Mun", "lat": "22.3933", "lng": "113.9745"},
    {"name": "Yuen_Long", "lat": "22.4445", "lng": "114.0228"},
    {"name": "North", "lat": "22.4975", "lng": "114.1366"},
    {"name": "Tai_Po", "lat": "22.4501", "lng": "114.1688"},
    {"name": "Sha_Tin", "lat": "22.3840", "lng": "114.1915"},
    {"name": "Sai_Kung", "lat": "22.3832", "lng": "114.2700"},
    {"name": "Kwai_Tsing", "lat": "22.3573", "lng": "114.1296"},
    {"name": "Islands", "lat": "22.2619", "lng": "113.9461"},
]

def fetch_all_restaurant_keyword(
    district_name: str,
    restaurant_name: str,
    latitude: str,
    longitude: str,
    max_retries: int = 3,
    delay: float = 1.0
) -> List[Dict]:
    """
    Get complete data for all branches of a specified restaurant (auto-pagination)
    
    Parameters:
        restaurant_name: Target restaurant name (e.g., "McDonald's")
        latitude/longitude: Search center coordinates
        max_retries: Maximum retry attempts
        delay: Request interval (seconds)
    
    Returns:
        List containing complete data for all branches (each entry includes district_name field)
    """
    all_keyword = []
    offset = 0
    total_expected = None
    session = requests.Session()
    request_count = 0

    # Configure request headers (simulate browser access)
    headers = {
        "x-disco-client-id": "web",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh;q=0.9"
    }

    while True:
        retries = max_retries
        success = False
        
        while retries > 0 and not success:
            try:
                url = (
                    f"https://hk.fd-api.com/vendors-gateway/api/v1/pandora/vendors"
                    f"?query={restaurant_name}"
                    f"&latitude={latitude}"
                    f"&longitude={longitude}"
                    f"&offset={offset}"
                    "&limit=50"  # Max items per page
                    "&opening_type=delivery"
                    "&vertical=restaurants"
                    "&country=hk"
                    "&locale=zh_HK"
                )

                response = session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                json_data = response.json()
                request_count += 1
                
                # Get total result count on first request
                if total_expected is None:
                    total_expected = json_data['data']['returned_count']
                    print(f"Found {restaurant_name} has {total_expected} stores in {district_name}")
                
                current_items = json_data['data']['items']
                
                if not current_items:
                    print("✅ Reached last page")
                    break

                # Exact name matching (prevent fuzzy matching) and add district_name field
                matched = [
                    {**branch, "district_name": district_name}  # Add district_name field
                    for branch in current_items
                    if restaurant_name.lower() in branch.get('name', '').lower() 
                ]
                #print(matched)
                
                all_keyword.extend(matched)
                print(f"📌 Page {offset//50 + 1} | Added {len(matched)} branches")

                # Termination condition check
                if len(current_items) < 50 or len(all_keyword) >= total_expected:
                    print("🏁 Data collection completed")
                    break

                offset += 50
                time.sleep(delay)
                success = True
                
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    print(f"❌ Failed to fetch data: {str(e)}")
                    break
                print(f"⚠️ Request error ({type(e).__name__}), {retries} retries left...")
                time.sleep(delay * 3)
        
        if not success or len(current_items) < 50 or len(all_keyword) >= total_expected:
            break

    # Result verification
    if total_expected is not None and len(all_keyword) != total_expected:
        print(f"⚠️ Warning: Retrieved {len(all_keyword)}/{total_expected} items")

    return all_keyword


import json
import os
from typing import List, Dict

def save_to_json(data: List[Dict], filename: str, append: bool = False):
    if not data:
        print("No data to save.")
        return

    # If in append mode and file exists, read existing data first
    if append and os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    data = existing_data + data  # Merge lists
                else:
                    data = [existing_data] + data  # Convert single object to list
            except json.JSONDecodeError:
                # If original file is not valid JSON, overwrite it
                pass

    # Unified write (overwrite or append merged data)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON saved to {filename}")

import csv
import os
from typing import List, Dict

def save_to_csv(data: List[Dict], filename: str):
    if not data:
        print("No data to save.")
        return

    file_exists = os.path.isfile(filename)
    
    with open(filename, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        #print(data)
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)
    print(f"CSV saved to {filename}")

import os
from typing import List, Dict, Optional

def save_to_txt(
    data: List[Dict], 
    filename: str, 
    fields: Optional[List[str]] = None,
    write_header: bool = True
):
    if not data:
        print("No data to save.")
        return

    default_fields = ["name", "address", "latitude", "longitude"]
    selected_fields = fields or default_fields
    file_exists = os.path.isfile(filename)

    with open(filename, "a" if file_exists else "w", encoding="utf-8") as f:
        # Write header (optional)
        if write_header and not file_exists:
            header = " | ".join(selected_fields)
            f.write(header + "\n")

        # Write data
        for item in data:
            line = " | ".join(str(item.get(field, "")) for field in selected_fields)
            f.write(line + "\n")
    
    print(f"TXT saved to {filename}")

def search_all_districts(keyword: str):
    """Search across all districts"""
    for district in DISTRICT_COORDS:
        district_name = district["name"]
        print(f"\n=== Searching in {district_name} ===")
        
        restaurants = fetch_all_restaurant_keyword(
            district["name"], keyword, district["lat"], district["lng"]
        )
        
        if restaurants:
            base_filename = f"{keyword}_{district_name}"
            #save_to_json(restaurants, f"{base_filename}.json")
            save_to_csv(restaurants, f"D:/HKBU/course/python/project/zxw/webscraping/output/totalStore.csv")
            save_to_txt(restaurants, f"D:/HKBU/course/python/project/zxw/webscraping/output/totalStore.txt")
            print(f"Total in {district_name}: {len(restaurants)}")
        else:
            print(f"No results in {district_name}")

if __name__ == "__main__":
    keyword = input("Enter keyword to search (e.g., Starbucks): ").strip()
    search_all_districts(keyword)
