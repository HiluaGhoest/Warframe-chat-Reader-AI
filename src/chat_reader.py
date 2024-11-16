import re
import json
import logging
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import sys
import os
import time
from fuzzywuzzy import process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Preprocess image to improve OCR accuracy
def preprocess_image(image_path):
    logging.info("Preprocessing the image...")
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Enhance contrast automatically
    image = ImageEnhance.Sharpness(image).enhance(2.0)  # Increase sharpness
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary thresholding
    logging.info("Image preprocessing complete.")
    return image

# Load the tradeable items from JSON
def load_tradeable_items():
    with open('weapons.json', 'r', encoding='utf-8') as f:
        weapons = json.load(f)

    with open('riven_ffixes.json', 'r', encoding='utf-8') as f:
        ffixes = json.load(f)

    combinations = ffixes.get("combinations", [])
    return weapons, ffixes, combinations  # Return combinations as well

def extract_valid_names(weapons, ffixes):
    """Extract valid names from weapons and ffixes for fuzzy matching."""
    valid_names = set(weapons)  # Assuming weapons is a list of valid weapon names
    # If ffixes contains valid prefixes or suffixes, add them here

    if "prefixes" in ffixes:
        valid_names.update(ffixes["prefixes"])

    if "suffixes" in ffixes:
        valid_names.update(ffixes["suffixes"])

    return list(valid_names)

def split_item_name(item_name):
    # Try to separate the prefix and suffix from the base item name
    parts = item_name.split('-')  # Adjust the separator if needed
    base_name = parts[0]
    prefix = "-".join(parts[1:]) if len(parts) > 1 else None
    suffix = None  # Implement logic to detect suffix if required
    
    return base_name, prefix, suffix


# Compare extracted items with the tradeable items list
def compare_with_tradeable_items(item_names, weapons, ffixes, combinations, trade_data):
    valid_items = []
    
    # Normalize weapon names to lower case and strip whitespace
    weapon_names = {weapon.lower().strip() for weapon in weapons}

    for item_name in item_names:
        result = extract_prefix_suffix(item_name)  # Extract prefix, suffix, and base name
        prefix = result.get("prefix", "")
        suffix = result.get("suffix", "")
        base_name = result["base_name"].strip()  # Ensure base_name is stripped of whitespace

        # Log the base name for debugging
        logging.info(f"Extracted base name: '{base_name}' from item name: '{item_name}'")

        # Check if the base name (without prefix/suffix) is in the tradeable items list
        item_found = base_name.lower() in weapon_names  # Check for base name

        # Construct the full item name for checking combinations
        full_item_name = f"{base_name}{prefix}{suffix}".lower()

        # Check for combined prefixes and suffixes
        combination_found = any(
            f"{base_name}{combined_suffix}".lower() in (combination.lower() for combination in combinations)
            for combined_suffix in (f"{prefix}{suffix}", f"{suffix}{prefix}")
        )

        if item_found and (prefix or suffix == "" or combination_found):
            # Find the corresponding trade entry for the item
            for entry in trade_data:
                for item in entry['items']:
                    if item['item'] == item_name:  # Ensure this matches the original item name
                        valid_items.append({
                            "username": entry['username'],  # Move username to the front
                            "action": entry['action'],  # Move action to the front
                            "item": item_name,  # Keep the full name (including prefix/suffix)
                            "weapon": base_name,
                            "prefix": prefix,
                            "core": base_name,  # Set core to the base name
                            "suffix": suffix,
                            "price": item.get('price')  # Include price
                        })
                        logging.info(f"Added item to valid items: {item_name}")
                        break  # Stop searching once the item is found
        else:
            logging.warning(f"Item {item_name} (base: {base_name}) is not tradable or not found.")
    
    return valid_items

def extract_prefix_suffix(item_name):
    """Extract prefix, suffix, and base name from the item name, adding space after weapon names."""
    prefix = ""
    suffix = ""
    base_name = ""

    # Load the weapon names from the JSON file
    weapons, _, combinations = load_tradeable_items()  # We only care about weapons and combinations here
    weapon_names = set(weapons)  # Convert to a set for fast lookup

    # Clean and prepare the item name
    cleaned_name = item_name.strip().lower()  # Convert to lowercase
    if cleaned_name.startswith('[') and cleaned_name.endswith(']'):
        cleaned_name = cleaned_name[1:-1].strip()  # Remove brackets and trim whitespace

    # Log the cleaned name for debugging
    logging.info(f"Cleaned name: '{cleaned_name}'")  # Log the cleaned name

    # Extract base name and check for weapon names
    for weapon in weapon_names:
        if cleaned_name.startswith(weapon):
            base_name = weapon  # Set base name to the weapon
            cleaned_name = cleaned_name[len(weapon):].strip()  # Remove weapon name from cleaned_name
            logging.info(f"Weapon found: '{weapon}', Remaining cleaned name: '{cleaned_name}'")
            break  # Assuming we only want to match the first weapon

    # If no weapon was found, the base name is the first part of the cleaned name
    if not base_name:
        base_name = cleaned_name.split()[0]  # First part is the base name
        cleaned_name = cleaned_name[len(base_name):].strip()  # Update cleaned_name

    # Remaining part after the base name
    remaining_part = cleaned_name.strip()

    # Check if the remaining part can be split into a valid combination of prefix and suffix
    for combination in combinations:
        if remaining_part.startswith(combination):
            # Split the combination into prefix and suffix
            prefix = combination[:-3]  # Assuming the suffix is always 3 characters long (e.g., 'tak')
            suffix = combination[-3:]  # Get the suffix
            logging.info(f"Combination found: Base name '{base_name}', Prefix '{prefix}', Suffix '{suffix}'")
            break

    # Check for hyphenated items only if remaining_part is not empty
    if remaining_part and '-' in remaining_part:
        parts = remaining_part.split('-')
        if len(parts) > 1:
            # If we have a hyphenated structure
            suffix = parts[-1]  # Last part after the last hyphen
            prefix = '-'.join(parts[:-1])  # Join all parts before the last hyphen as prefix
            logging.info(f"Hyphenated found: Base name '{base_name}', Prefix '{prefix}', Suffix '{suffix}'")
        else:
            # If there's no hyphen, we don't need to change anything
            logging.info(f"No hyphenated structure found in remaining part: '{remaining_part}'")

    # Log the extracted values for debugging
    logging.info(f"Extracted base name: '{base_name}', Prefix found: '{prefix}', Suffix found: '{suffix}'")

    # Prepare the result dictionary
    result = {"base_name": base_name}
    
    # Only include prefix and suffix if they are found (not empty)
    if prefix:
        result["prefix"] = prefix
    if suffix:
        result["suffix"] = suffix
    
    return result


def parse_trade_data(raw_text):
    # Extract username, action, items, and prices from the raw text
    match = re.match(r"(\w+):+(wts|wtb|wtt)+(.*)", raw_text.strip(), re.IGNORECASE)
    if not match:
        return []

    username = match.group(1)
    action = match.group(2).upper()  # Normalize action to uppercase
    items_raw = match.group(3).strip()  # The raw items part

    # Extract individual items and their optional prices
    item_price_pairs = re.findall(r'\[(.*?)\](\d+)?', items_raw)  # Find items and optional prices
    trade_data = {
        'username': username,
        'action': action,
        'items': []
    }

    # Default currency
    default_currency = "Platinum"

    # Initialize variable to hold the current price
    current_price = None
    null_price_items = []  # List to hold items with null prices

    # Iterate over item-price pairs
    for index, (item, price) in enumerate(item_price_pairs):
        # Clean the item name and extract prefix, suffix, and base name
        result = extract_prefix_suffix(item)
        prefix = result.get("prefix", "")
        suffix = result.get("suffix", "")
        base_name = result["base_name"]

        # Determine the price to assign
        if price:
            current_price = price  # Update current price
            # Assign the current price to all previously stored items with null prices
            for null_item in null_price_items:
                null_item['price'] = current_price  # Set the stored null price to the current price
                # Add the null item to trade_data before clearing the list
                trade_data['items'].append(null_item)
            null_price_items.clear()  # Clear the list after assigning prices
            
            # Now add the current item with its price
            item_entry = {
                'item': item.strip(),
                'price': current_price,  # Assign the price (current)
                'currency': default_currency,
                'is_discussable': "true"
            }
            # Only include prefix if it's not empty
            if prefix:
                item_entry['prefix'] = prefix
            # Only include suffix if it's not empty
            if suffix:
                item_entry['suffix'] = suffix
            
            trade_data['items'].append(item_entry)
        else:
            # Store the item with null price for later assignment
            null_price_items.append({
                'item': item.strip(),
                'prefix': prefix,
                'suffix': suffix,
                'price': None,  # Initially set price as None
                'currency': default_currency,
                'is_discussable': "true"
            })

    # If there are any remaining null price items after the loop, they will remain with None price
    for null_item in null_price_items:
        # Check if both prefix and suffix are empty before adding
        if null_item['prefix'] or null_item['suffix']:
            trade_data['items'].append(null_item)

    return [trade_data]  # Return a list containing the trade data entry
    
def normalize_ocr_output(ocr_text):
    # Log the raw OCR output
    logging.info("Raw OCR output: %s", ocr_text)

    # Split the OCR output into individual trades using a regex that captures the username and action
    trades = re.findall(r'(\w+)\s*:\s*(wts|wtb|wtt)(.*?)(?=\s+\w+\s*:|$)', ocr_text, re.IGNORECASE | re.DOTALL)
    
    # trades will now be a list of tuples (username, action, items)
    print("Extracted trades:", trades)

    # Further refine the trades to ensure proper format
    refined_trades = []
    
    for username, action, items_raw in trades:
        username = username.strip()
        action = action.upper()  # Normalize action to uppercase
        items_raw = items_raw.strip()
        
        # Refine items to ensure they are properly formatted
        items = re.findall(r'\[([^\]]+?)\](\d*)', items_raw)  # Find items with optional prices
        items_string = ''.join(f'[{item}]{price}' for item, price in items)  # Join items into a single string
        
        # Construct the normalized trade string
        if items_string:  # Only add if items_string is not empty
            refined_trade = f"{username}:{action}{items_string}"
            refined_trades.append(refined_trade)
            print("Refined trade:", refined_trade)

    print("Final refined trades:", refined_trades)
    return refined_trades

def correct_item_names(item_data, valid_names, valid_prefixes, valid_suffixes):
    """Correct the item names using fuzzy matching and handle core extraction."""
    for item in item_data:
        item_name = item["item"]
        prefix = item["prefix"]
        suffix = item["suffix"]
        core = ""  # Initialize core as an empty string

        # Check if the suffix matches any valid prefixes
        for valid_prefix in valid_prefixes:
            if suffix.lower().startswith(valid_prefix.lower()):
                core = valid_prefix  # Set core to the matched prefix
                suffix = suffix[len(valid_prefix):].strip()  # Remove the prefix from the suffix
                print(f"Extracting Core: '{core}', Remaining Suffix: '{suffix}' for item '{item_name}'")
                break  # Exit after the first valid prefix match

        # Check if the suffix is correct after core extraction
        if suffix.lower() not in valid_suffixes:
            # Find the closest match for the suffix
            closest_match, score = process.extractOne(suffix, valid_suffixes)
            # If the score is above a certain threshold, replace the suffix
            if score >= 80:
                print(f"Correcting '{suffix}' to '{closest_match}' for item '{item_name}'")
                suffix = closest_match

        # Split the suffix into prefix and suffix if applicable
        for valid_suffix in valid_suffixes:
            if suffix.lower().endswith(valid_suffix):
                # Determine the prefix part
                prefix_candidate = suffix[:-len(valid_suffix)].strip()
                if prefix_candidate in valid_prefixes:
                    prefix = prefix_candidate
                    suffix = valid_suffix
                    print(f"Splitting '{item_name}' into Prefix: '{prefix}', Suffix: '{suffix}'")
                break  # Exit after first valid split

        # Optionally, check the prefix as well
        if prefix.lower() not in valid_prefixes:
            closest_match, score = process.extractOne(prefix, valid_prefixes)
            if score >= 80:
                print(f"Correcting '{prefix}' to '{closest_match}' for item '{item_name}'")
                prefix = closest_match
        
        # Update the item dictionary
        item["core"] = core  # Add core key to the item dictionary
        item["prefix"] = prefix  # Update prefix
        item["suffix"] = suffix  # Update suffix
    
    return item_data

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_from_json(filename):
    """Load data from a JSON file, handling cases where the file may not exist or is malformed."""
    if not os.path.exists(filename):
        return []  # Return an empty list if the file does not exist
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"Warning: The file '{filename}' is malformed. Starting fresh.")
        return []  # Return an empty list if the file is malformed

def filter_trade_data(trade_data):
    """Filter trade data based on specified criteria and remove duplicates."""
    filtered_data = []
    seen_items = set()  # To track unique combinations of (username, item, prefix, core, suffix, price)
    
    for entry in trade_data:
        # Normalize fields for comparison, ensuring we handle None values
        username = (entry['username'] or '').strip().lower()  # Default to empty string if None
        item = (entry['item'] or '').strip().lower()  # Default to empty string if None
        prefix = (entry.get('prefix') or '').strip().lower()  # Default to empty string if None
        core = (entry.get('core') or '').strip().lower()  # Default to empty string if None
        suffix = (entry.get('suffix') or '').strip().lower()  # Default to empty string if None
        price = entry.get('price', '')

        # Check if the price is valid (not null and a number)
        if price is None or not isinstance(price, (int, str)) or (isinstance(price, str) and not price.isdigit()):
            continue
        
        # Convert price to an integer for further processing
        price = int(price)

        # Check if the item does not contain unwanted patterns
        if 'wts' in item:
            continue
        
        # Create a unique key based on multiple fields
        unique_key = (username, item, prefix, core, suffix, price)
        
        # Check for duplicates based on the unique key
        if unique_key in seen_items:
            continue
        seen_items.add(unique_key)
        
        # If it passes all checks, add it to the filtered list
        filtered_data.append(entry)
    
    logging.info(f"Filtered data size: {len(filtered_data)}")
    return filtered_data

# Main function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chat_reader.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]  # Get the image path from the command line
    start_time = time.time()
    logging.info("Script started.")

    try:
        # Preprocess image
        preprocessed_image = preprocess_image(image_path)

        # Perform OCR
        logging.info("Performing OCR...")
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?\'\"-_:[]{}() '
        raw_text = pytesseract.image_to_string(preprocessed_image, config=custom_config).lower()
        raw_text = re.sub(r'\s+', ' ', raw_text)

        logging.info("OCR complete.")

        # Normalize trade data
        normalized_trades = normalize_ocr_output(raw_text)
        logging.info("Normalized trades: %s", normalized_trades)

        # Initialize trade_data as an empty list
        trade_data = []

        # Parse each normalized trade
        if normalized_trades:
            for trade in normalized_trades:
                parsed_data = parse_trade_data(trade)
                logging.info(f"Parsed trade data: {parsed_data}")
                trade_data.extend(parsed_data)  # Extend the trade_data list with parsed data
        else:
            logging.warning("No normalized trades found.")

        # Load the tradeable items and combinations
        weapons, ffixes, combinations = load_tradeable_items()

        # Extract valid names for correction
        valid_names = extract_valid_names(weapons, ffixes)

        # Extract item names from the parsed trade data
        item_names = []
        for entry in trade_data:
            for item in entry['items']:
                item_names.append(item['item'])

        # Compare extracted items with the tradeable items list
        valid_items = compare_with_tradeable_items(item_names, weapons, ffixes, combinations, trade_data)

        # Correct the item names
        corrected_items = correct_item_names(valid_items, valid_names, ffixes["prefixes"], ffixes["suffixes"])

        logging.info(f"Extracted {len(corrected_items)} valid tradeable items after correction.")
        for entry in corrected_items:
            logging.info(f"Valid item: {entry['item']}, Prefix: {entry['prefix']}, Suffix: {entry['suffix']}")

        
        # Step 1: Save the parsed data to a temporary JSON file
        temp_filename = 'temp_trade_data.json'
        save_to_json(corrected_items, temp_filename)

        # Step 2: Load the JSON data from the temporary file
        trade_data = load_from_json(temp_filename)

        # Step 3: Filter the trade data
        filtered_trade_data = filter_trade_data(trade_data)

        # Step 4: Append the filtered data to another JSON file
        output_filename = 'final_trade_data.json'

        # Load existing data from the output file if it exists
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                final_data = load_from_json(output_filename)
        else:
            final_data = []

        # Create a set of existing unique keys to avoid duplicates
        existing_keys = set((entry['username'], entry['item'], entry['prefix'], entry['core'], entry['suffix'], entry['price']) for entry in final_data)

        # Append the filtered data while checking for duplicates
        for new_entry in filtered_trade_data:
            # Normalize fields for comparison
            new_username = new_entry['username'].strip().lower()
            new_item = new_entry['item'].strip().lower()
            new_prefix = new_entry.get('prefix', '').strip().lower()
            new_core = new_entry.get('core', '').strip().lower()
            new_suffix = new_entry.get('suffix', '').strip().lower()
            new_price = new_entry.get('price', '').strip()

            # Create a unique key for the new entry
            new_key = (new_username, new_item, new_prefix, new_core, new_suffix, new_price)

            # Only append if the new key is not in the existing keys
            if new_key not in existing_keys:
                final_data.append(new_entry)
                existing_keys.add(new_key)  # Add the new key to the set

        # Save the updated data back to the output file
        with open(output_filename, 'w') as f:
            json.dump(final_data, f, indent=4)

        print("Filtered trade data has been successfully saved to", output_filename)

        # Append the filtered data
        final_data.extend(filtered_trade_data)

        # Save the updated data back to the output file
        with open(output_filename, 'w') as f:
            json.dump(final_data, f, indent=4)

        # Clean up: Remove the temporary file
        os.remove(temp_filename)
        os.remove(image_path)

        print("Filtered trade data has been successfully saved to", output_filename)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    logging.info(f"Execution time: {time.time() - start_time:.2f} seconds.")