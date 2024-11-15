import cv2
import numpy as np
import pyautogui
import time
import os
from multiprocessing import Process, Queue, Pool
import cProfile
import pstats
import subprocess

def init_opencl():
    """Initialize OpenCL context"""
    try:
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            print("OpenCL is available")
            print(f"OpenCL device: {cv2.ocl.Device.getDefault().name()}")
            return True
        else:
            print("OpenCL is not available")
            return False
    except Exception as e:
        print(f"OpenCL initialization failed: {e}")
        return False

def preprocess_image_opencl(image):
    """OpenCL-accelerated image preprocessing"""
    gpu_image = cv2.UMat(image)
    gpu_gray = cv2.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
    gpu_blur = cv2.GaussianBlur(gpu_gray, (3, 3), 0)
    edges = cv2.Canny(gpu_blur, 30, 100)
    kernel = np.ones((2, 2), np.uint8)
    gpu_dilated = cv2.dilate(edges, kernel)
    return gpu_dilated

def calculate_correction(scale):
    """Calculate correction factor and offset based on linear interpolation between 0%, 100%, and 140%,
       and extrapolate for scales above 140%."""
    
    # Known correction values at 50%, 100%, and 140%
    correction_50 = 2.4  
    x_offset_50 = -40    
    y_offset_50 = 0      

    correction_100 = 1.0
    x_offset_100 = -20
    y_offset_100 = 0

    correction_140 = 0.7
    x_offset_140 = 0
    y_offset_140 = 0

    # If scale is exactly 0%, 100%, or 140%
    if scale == 50:
        return correction_50, x_offset_50, y_offset_50
    elif scale == 100:
        return correction_100, x_offset_100, y_offset_100
    elif scale == 140:
        return correction_140, x_offset_140, y_offset_140

    # Extrapolation for scales below 100%
    if scale < 100:
        # Calculate the scale fraction for extrapolation
        scale_fraction = (scale - 50) / (100 - 50)
        
        # Interpolated values for below 100%
        correction_factor = correction_50 + scale_fraction * (correction_100 - correction_50)
        x_offset = x_offset_50 + scale_fraction * (x_offset_100 - x_offset_50)
        y_offset = y_offset_50 + scale_fraction * (y_offset_100 - y_offset_50)
        
        return correction_factor, int(x_offset), int(y_offset)

    # Linear interpolation between 100% and 140%
    if scale <= 140:
        scale_fraction = (scale - 100) / (140 - 100)
        
        # Interpolated values for between 100% and 140%
        correction_factor = correction_100 + scale_fraction * (correction_140 - correction_100)
        x_offset = x_offset_100 + scale_fraction * (x_offset_140 - x_offset_100)
        y_offset = y_offset_100 + scale_fraction * (y_offset_140 - y_offset_100)

        return correction_factor, int(x_offset), int(y_offset)

    # Extrapolation for scales above 140%
    if scale > 140:
        # Calculate the scale fraction for extrapolation above 140%
        scale_fraction = (scale - 140) / (160 - 140)  # Using 160 as a hypothetical reference for extrapolation
        
        # Calculate extrapolated values based on the trend from 100% to 140%
        correction_factor = correction_140 + scale_fraction * (correction_140 - correction_100)
        x_offset = x_offset_140 + scale_fraction * (x_offset_140 - x_offset_100)
        y_offset = y_offset_140 + scale_fraction * (y_offset_140 - y_offset_100)

        return correction_factor, int(x_offset), int(y_offset)

def load_templates(template_paths):
    """Load multiple templates and prepare for caching scaled images."""
    templates = {}
    for name, path in template_paths.items():
        template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if template is None:
            print(f"Failed to load template: {path}. Please check the file path and ensure the file exists.")
            continue
        
        # Check if the image has an alpha channel
        if template.shape[-1] == 4:
            # Separate the alpha channel
            b, g, r, alpha = cv2.split(template)
            mask = alpha > 0  # Create mask from alpha channel
            template = cv2.merge((b, g, r))  # Remove alpha for template matching
        else:
            mask = None  # No mask needed if there's no transparency
        
        templates[name] = {
            'image': template,
            'mask': mask,
            'height': template.shape[0],
            'width': template.shape[1],
        }
    return templates

def scale_template(template, scales):
    """Scale the template to various sizes and return a dictionary of scaled images."""
    scaled_templates = {}
    for scale in scales:
        scaled_image = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_templates[scale] = scaled_image
    return scaled_templates

def load_template_with_alpha(path):
    """Load a template image with transparency."""
    template = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    if template is None:
        print(f"Failed to load template: {path}")
        return None

    # Split the template into its color and alpha channels
    b, g, r, alpha = cv2.split(template)

    # Create a mask using the alpha channel
    mask = alpha > 0  # Create a binary mask where the alpha channel is greater than 0
    return cv2.merge((b, g, r)), mask  # Return the BGR image and mask


def find_fixed_images(templates, screenshot, scales, threshold=0.3, early_stop_threshold=0.9):
    """Locate fixed images in the screenshot for one specific template using cached scaled images."""
    matches = {}
    cache = {}  # Cache for scaled images
    
    for name, template_data in templates.items():
        template = template_data['image']
        mask = template_data['mask']
        
        # Pre-load scaled images for the current template if not cached
        if name not in cache:
            cache[name] = scale_template(template, scales)

        for scale in scales:
            resized_template = cache[name][scale]
            resized_mask = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Perform template matching using the resized template
            result = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED, mask=resized_mask)
            locations = np.where(result >= threshold)

            if locations[0].size > 0:
                for y, x in zip(locations[0], locations[1]):
                    score = result[y, x]
                    reliability = score * (np.sum(mask) / mask.size if mask is not None else 1)

                    # Check if reliability is valid
                    if reliability != float('inf') and reliability == reliability:
                        matches.setdefault(name, []).append((x, y, scale, reliability)) 

                        # Early stopping condition based on reliability
                        if reliability >= early_stop_threshold:  
                            print(f"Early stopping for {name} at scale {scale} with reliability {reliability:.2f}")
                            break

                # If a sufficient match has been found, we can break out of the scale loop early
                if name in matches and any(m[3] >= early_stop_threshold for m in matches[name]):
                    break

    return matches

def worker(template_data, screenshot, scales, output_queue):
    """Worker function for processing template matching."""
    matches = {}
    # Process the screenshot with the provided template data and scales
    matches.update(find_fixed_images(template_data, screenshot, scales))
    # Send the results back to the output queue
    output_queue.put(matches)



def calculate_chat_boundaries(matches):
    """Calculate the bounding rectangle for the chat window based on found image positions."""
    if len(matches) != 2:
        print("Both fixed images must be found to calculate boundaries.")
        return None

    # Ensure you are getting the correct tuple
    top_left = matches['chat_window_top_left']  # Get the best match for top_left
    bottom_right = matches['chat_window_bottom_right']  # Get the best match for bottom_right

    # Ensure top_left and bottom_right are tuples with at least 4 elements
    if len(top_left) < 4 or len(bottom_right) < 4:
        print("Error: Matches do not have the expected format.")
        return None

    # Calculate the rectangle coordinates
    x1, y1 = top_left[0], top_left[1]
    x2 = bottom_right[0]
    y2 = bottom_right[1]

    return (x1, y1, x2 - x1, y2 - y1)  # (x, y, width, height)



def calculate_combined_rectangle(top_left, bottom_right):
    """Calculate a new rectangle based on top_left and bottom_right coordinates."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Create the new rectangle (x, y, width, height)
    combined_rectangle = (x1, y1, width, height)
    
    return combined_rectangle


def main():
    if not init_opencl():
        print("OpenCL acceleration not available. Exiting...")
        return

    # Define the paths for templates
    template_paths = {
        'chat_window_top_left': 'templates/top_left.png',
        'chat_window_bottom_right': 'templates/bottom_right.png',
    }
    
    # Load templates
    templates = load_templates(template_paths)
    scales = np.linspace(0.75, 2.0, num=7)  # Scales for template matching
    output_path = 'screenshots/chat_detection_output.png'  # Ensure this path is writable

    # Initialize output queue
    output_queue = Queue()
    print("Starting multi-scale detection...")

    # Create a pool of processes upfront
    processes = []
    
    try:
        # Capture screenshot once
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Create and start processes for each template
        for name in templates:
            p = Process(target=worker, args=({name: templates[name]}, screenshot, scales, output_queue))
            processes.append(p)
            p.start()

        # Collect results from the output queue
        all_matches = {}
        for _ in processes:
            matches = output_queue.get()  # Collect results from each worker
            all_matches.update(matches)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Clear the processes list
        processes.clear()

        if all_matches:
            best_matches = {}

            # Find the best match for bottom_right and top_left
            for name, found_matches in all_matches.items():
                if name in ['chat_window_bottom_right', 'chat_window_top_left']:
                    found_matches.sort(key=lambda x: x[3], reverse=True)  # Sort by reliability
                    best_match = found_matches[0]  # Get the best match
                    best_matches[name] = best_match  # Store the best match

            # Calculate new rectangles using the best matches
            if 'chat_window_top_left' in best_matches and 'chat_window_bottom_right' in best_matches:

                top_x, top_y, top_scale, top_reliability = best_matches['chat_window_top_left']
                top_width = int(templates['chat_window_top_left']['width'] * top_scale)
                top_height = int(templates['chat_window_top_left']['height'] * top_scale)

                bottom_x, bottom_y, bottom_scale, bottom_reliability = best_matches['chat_window_bottom_right']
                bottom_width = int(templates['chat_window_bottom_right']['width'] * bottom_scale)
                bottom_height = int(templates['chat_window_bottom_right']['height'] * bottom_scale)

                x1_chat = top_x
                y1_chat = top_y
                x2_chat = (bottom_x + bottom_width)
                y2_chat = (bottom_y + bottom_height)

                cv2.rectangle(screenshot, (x1_chat - top_width, y1_chat - top_height), (x2_chat + bottom_width, y2_chat + bottom_height), (0, 255, 0), 2) # outer area

                x1_text = top_x
                y1_text = (top_y + top_height)
                x2_text = (bottom_x + bottom_width)
                y2_text = bottom_y

                cv2.rectangle(screenshot, (x1_text, y1_text), (x2_text, y2_text), (0, 0, 255), 2) # inter area

                        
                cv2.rectangle(screenshot, (top_x, top_y), ((top_x + top_width), (top_y + top_height)), (255, 255, 0), 2)
                cv2.rectangle(screenshot, (bottom_x, bottom_y), ((bottom_x + bottom_width), (bottom_y + bottom_height)), (255, 255, 0), 2)

                # Call the second script for capture
                import subprocess

                # Rectangle bounds as a string
                bounds_args = f"{x1_chat - top_width},{y1_chat - top_height},{x2_chat + bottom_width},{y2_chat + bottom_height} " \
                            f"{x1_text},{y1_text},{x2_text},{y2_text}"

                # Call another script with bounds as arguments
                subprocess.run(['python', 'screenshot_capture.py'] + bounds_args.split())

                cv2.imwrite('screenshots/chat_detection_output.png', screenshot)
            else:
                print(f"Failed to save the output image to {output_path}")
        else:
            print("No matches found.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    finally:
        # Ensure all processes are properly terminated
        for p in processes:
            p.join()  # Wait for all processes to finish

if __name__ == "__main__":
    # Create a Profile object
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    try:
        main()  # Run main function
    finally:
        profiler.disable()  # Stop profiling
        # Save profiling results to a file
        profiler.dump_stats("profile_results.prof")

    # Optionally, you can print stats to the console
    with open("profile_output.txt", "w") as f:
        ps = pstats.Stats("profile_results.prof", stream=f)
        ps.sort_stats("cumulative")  # You can sort by different criteria, e.g., "time", "cumulative"
        ps.print_stats()