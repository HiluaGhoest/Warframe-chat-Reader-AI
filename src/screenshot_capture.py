import sys
import pyautogui
from PIL import Image
import cv2
import numpy as np
import subprocess
import time
import threading
import pygetwindow as gw

# Function to run the chat reader script asynchronously
def run_chat_reader(image_path):
    print(f"Starting chat reader for {image_path}...")
    subprocess.call(['python', 'chat_reader.py', image_path])
    print(f"Chat reader finished for {image_path}.")

# Function to start chat reader with a delay
def delayed_chat_reader(image_path, delay=1):
    time.sleep(delay)  # Add delay before running the subprocess
    run_chat_reader(image_path)

# Function to perform the template detection and capture screenshots
def detect_and_capture(outer_bounds, inner_bounds, threshold=0.5):
    # Get the active window title
    active_window = gw.getActiveWindow()
    
    # Check if Warframe is focused (replace 'Warframe' with the actual window title)
    if active_window and 'Warframe' in active_window.title:
        print("Warframe is focused. Proceeding with template detection...")

        # Define the bounds of the outer area
        x1, y1, x2, y2 = outer_bounds

        # Capture the screenshot within the outer bounds
        screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))

        # Convert the screenshot to a format usable by OpenCV
        screenshot_np = np.array(screenshot)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

        # Load template images
        template1 = cv2.imread('templates/top_left.png', cv2.IMREAD_GRAYSCALE)
        template2 = cv2.imread('templates/bottom_right.png', cv2.IMREAD_GRAYSCALE)

        # Perform template matching for the first template
        result1 = cv2.matchTemplate(screenshot_gray, template1, cv2.TM_CCOEFF_NORMED)
        _, max_val1, _, max_loc1 = cv2.minMaxLoc(result1)

        # Perform template matching for the second template
        result2 = cv2.matchTemplate(screenshot_gray, template2, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, max_loc2 = cv2.minMaxLoc(result2)

        # Check if templates match within the threshold
        template1_found = max_val1 >= threshold
        template2_found = max_val2 >= threshold

        if template1_found or template2_found:
            if template1_found:
                print(f"Template 1 found at {max_loc1} with confidence {max_val1:.2f}")
            if template2_found:
                print(f"Template 2 found at {max_loc2} with confidence {max_val2:.2f}")

            # Capture the inner area screenshot
            x1_inner, y1_inner, x2_inner, y2_inner = inner_bounds
            inner_screenshot = pyautogui.screenshot(region=(x1_inner, y1_inner, x2_inner - x1_inner, y2_inner - y1_inner))

            # Convert to a format usable by OpenCV
            inner_screenshot_np = np.array(inner_screenshot)

            # Save the screenshot
            image_path = 'screenshots/inner_area_screenshot.png'
            cv2.imwrite(image_path, cv2.cvtColor(inner_screenshot_np, cv2.COLOR_RGB2BGR))
            print(f"Inner area screenshot saved as '{image_path}'.")

            # Run the chat reader in a separate thread
            thread = threading.Thread(target=delayed_chat_reader, args=(image_path, 1))  # Add a 1-second delay
            thread.start()
        else:
            print("No template detected in the outer area. No inner area screenshot captured.")
    else:
        return

# Convert the comma-separated strings into lists of integers
outer_bounds = list(map(int, sys.argv[1].split(',')))
inner_bounds = list(map(int, sys.argv[2].split(',')))

# Main loop
try:
    while True:
        detect_and_capture(outer_bounds, inner_bounds, threshold=0.5)
        time.sleep(1)  # Wait for 1 second before the next iteration
except KeyboardInterrupt:
    print("Script terminated by user.")
