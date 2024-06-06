# Basic UI

# import dependencies
import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
import cv2
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.image as mpimg
import matplotlib as mpl
import os
import zipfile
from collections import defaultdict
import random
import re
import logging

# Set the logging level to suppress debug and warning messages from PaddleOCR
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Setup the page
st.set_page_config(page_icon="📄👁️", layout="wide", initial_sidebar_state="expanded")

# Function to filter vertical lines
def vertical_detector(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    kernel_length = np.array(img).shape[1] // 40
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    img_final_bin = cv2.erode(~vertical_lines_img, kernel, iterations=2)
    _, img_final_bin = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_final_bin

# Function to detect the two vertical lines of the outline
def detect_and_draw_v_lines(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)
    points = []
    if lines is not None:
        a, _, _ = lines.shape
        for i in range(a):
            x1, y1, x2, y2 = lines[i][0]
            points.append((x1, y1))
            points.append((x2, y2))
    threshold = 5
    groups = defaultdict(list)
    for x, y in points:
        aligned = False
        for key in groups.keys():
            if abs(x - key) <= threshold:
                groups[key].append((x, y))
                aligned = True
                break
        if not aligned:
            groups[x].append((x, y))
    single_lines = []
    for key, group in groups.items():
        xs = [x for x, y in group]
        ys = [y for x, y in group]
        avg_x = int(np.mean(xs))
        min_y = min(ys)
        max_y = max(ys)
        length = max_y - min_y
        if length > 0.25 * height:
            single_lines.append((avg_x, min_y, avg_x, max_y, length))
    if len(single_lines) >= 2:
        single_lines.sort()
        max_distance = 0
        line1 = None
        line2 = None
        for i in range(len(single_lines) - 1):
            for j in range(i + 1, len(single_lines)):
                distance = abs(single_lines[j][0] - single_lines[i][0])
                if distance > max_distance:
                    max_distance = distance
                    line1 = single_lines[i]
                    line2 = single_lines[j]

    if line1[0] > line2[0]:
        line1, line2 = line2, line1
    image_width = gray.shape[1]
    if line1[0] / image_width > 0.35:
        line1 = (0, line1[1], 0, line1[3])
    if (image_width - line2[0]) / image_width > 0.35:
        line2 = (image_width, line2[1], image_width, line2[3])
    vertical_lines = [line1[0], line2[0]]
    return vertical_lines

# Function to filter horizontal lines
def horizontal_detector(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    kernel_length = np.array(img).shape[1] // 40
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    img_final_bin = cv2.erode(~horizontal_lines_img, kernel, iterations=2)
    _, img_final_bin = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_final_bin

# Function to detect the two horizontal lines of the outline
def detect_and_draw_h_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)
    if lines is None:
        print("No lines detected")
        return
    points = []
    a, _, _ = lines.shape
    for i in range(a):
        x1, y1, x2, y2 = lines[i][0]
        points.append((x1, y1))
        points.append((x2, y2))
    threshold = 5
    groups = defaultdict(list)
    for x, y in points:
        aligned = False
        for key in groups.keys():
            if abs(y - key) <= threshold:
                groups[key].append((x, y))
                aligned = True
                break
        if not aligned:
            groups[y].append((x, y))
    document_width = image.shape[1]
    min_line_length = 0.25 * document_width
    constructed_lines = []
    for key, group in groups.items():
        avg_y = int(np.mean([y for x, y in group]))
        min_x = min([x for x, y in group])
        max_x = max([x for x, y in group])
        line_length = max_x - min_x
        if line_length >= min_line_length:
            constructed_lines.append((min_x, avg_y, max_x, avg_y))
    max_distance = 0
    line1 = None
    line2 = None
    for i in range(len(constructed_lines)):
        for j in range(i + 1, len(constructed_lines)):
            _, y1, _, _ = constructed_lines[i]
            _, y2, _, _ = constructed_lines[j]
            distance = abs(y1 - y2)
            if distance > max_distance:
                max_distance = distance
                line1 = constructed_lines[i]
                line2 = constructed_lines[j]

    if line1[1] > line2[1]:
        line1, line2 = line2, line1
    if line1[1] > 150:
        line1 = (line1[0], 0, line1[2], 0)
    image_height = image.shape[0]
    if image_height - line2[1] > 150:
        line2 = (line2[0], image_height, line2[2], image_height)
    horizontal_lines = [line1[1], line2[1]]
    return horizontal_lines

# Function to crop the image through the outline and return it
def create_image_with_lines(image_path, vertical_lines, horizontal_lines):
    image = Image.open(image_path)
    width, height = image.size
    #print("Original Image size:", image.size)

    # Ensure we have exactly two vertical and two horizontal lines
    if len(vertical_lines) != 2 or len(horizontal_lines) != 2:
        raise ValueError("Expected exactly two vertical and two horizontal lines.")

    # Sort the lines to determine the cropping boundaries
    vertical_lines.sort()
    horizontal_lines.sort()

    left = vertical_lines[0]
    right = vertical_lines[1]
    top = horizontal_lines[0]
    bottom = horizontal_lines[1]

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    #print("Cropped image size:", cropped_image.size)

    return cropped_image

# Construct a single pipeline upto cropping throgh the outline
def outline_detection_pipeline(image_path):
    vertical_lines_image = vertical_detector(image_path)
    vertical_lines_image = cv2.cvtColor(vertical_lines_image, cv2.COLOR_GRAY2BGR)
    vertical_lines = detect_and_draw_v_lines(vertical_lines_image)
    #print("The X Co-Ordianates of Vertical lines to crop:", vertical_lines)

    horizontal_lines_image = horizontal_detector(image_path)
    horizontal_lines_image = cv2.cvtColor(horizontal_lines_image, cv2.COLOR_GRAY2BGR)
    horizontal_lines = detect_and_draw_h_lines(horizontal_lines_image)
    #print("The Y Co-Ordinates of Horizontal lines to crop:", horizontal_lines)

    cropped_image = create_image_with_lines(image_path, vertical_lines, horizontal_lines)
    return cropped_image

#############
# Function to perform OCR on cropped image
def perform_ocr_on_cropped_image(cropped_image, ocr_coords_percentage, threshold_percentage=0):
    cropped_rgb = np.array(cropped_image.convert('RGB'))
    height, width, _ = cropped_rgb.shape

    # Calculate top left and bottom right coordinates with threshold
    top_left = (int((ocr_coords_percentage[0] - threshold_percentage) * width / 100),
                int((ocr_coords_percentage[1] - threshold_percentage) * height / 100))
    bottom_right = (int((ocr_coords_percentage[2] + threshold_percentage) * width / 100),
                    int((ocr_coords_percentage[3] + threshold_percentage) * height / 100))

    # Ensure coordinates are within image boundaries
    top_left = (max(0, top_left[0]), max(0, top_left[1]))
    bottom_right = (min(width, bottom_right[0]), min(height, bottom_right[1]))

    #print("top_left Co-Ordinate of OCR zone on cropped image:", top_left[0], top_left[1])
    #print("bottom_right Co-Ordinate of OCR zone on cropped image:", bottom_right[0], bottom_right[1])

    # Crop the region for OCR
    ocr_area = cropped_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Perform OCR using PaddleOCR
    ocr = PaddleOCR(lang='en')
    result = ocr.ocr(ocr_area, cls=True)

    if result is None or len(result) == 0:
        print("No text detected in the specified area.")
        return
    #Display the OCR area
    # plt.imshow(ocr_area)
    # plt.title("Selected OCR Area")
    # plt.show()

    return result



def process_ocr_result_reg_no(result):
    regex_pattern = r'(?:[a-zA-Z]{2,3}-?\d{4}|[0-9]+-[0-9]+)$' # minimum 2/3 letters then "-" or without "-" then 4 four digits
    detected_texts = []

    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            if re.search(regex_pattern, text):
                detected_text = text
                detected_texts.append(detected_text)

    if len(detected_texts) == 0:
        return "Oops, No pattern match text found"  # Handle the case where no matching text is found

    # List of prefixes to check for
    prefixes = ["WP", "CP", "SP", "NP", "EP", "NW", "NC", "SG", "UW", "80"]

    for text in detected_texts:
        modified_text = text  # Start with the original text

        # Check if the text starts with any of the specified prefixes
        for prefix in prefixes:
            if text.startswith(prefix):
                # Only add a space if there isn't one already after the prefix
                if not text[len(prefix):].startswith(" "):
                    modified_text = prefix + " " + text[len(prefix):]
                break  # No need to check other prefixes once a match is found

        REGISTRATION_NO = modified_text
        return REGISTRATION_NO  # Return the first valid registration number found

    return "Oops, No pattern match text found"  # Return a message if no valid text is found

def process_ocr_result_chassis_no(result):
    # Define regex patterns
    patterns = [
        r'^(?=.*[A-Z].*[A-Z])(?=.*[0-9].*[0-9])[A-Za-z0-9]{17}$',  # 17 character string with at least 2 capital letters and 2 numbers
        r'^(?=.*[A-Z].*[A-Z])(?=.*[0-9].*[0-9])[A-Za-z0-9]{16}$',  # 16 character string with at least 2 capital letters and 2 numbers
        r'^(?=.*[A-Z].*[A-Z])(?=.*[0-9].*[0-9])[A-Za-z0-9-]{13}$',  # 13 character string with at least 2 capital letters, 2 numbers and a '-'
        r'^(?=.*[A-Z].*[A-Z])(?=.*[0-9].*[0-9])[A-Za-z0-9-]{14}$',  # 14 character string with at least 2 capital letters, 2 numbers and a '-'
        r'^(?=.*[A-Z].*[A-Z])(?=.*[0-9].*[0-9])[A-Za-z0-9-]{18}$',  # 18 character string with at least 2 capital letters, 2 numbers and a '-'
        r'^[A-Za-z]{17}$'  # 17 character string with all letters
    ]

    detected_texts = []
    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            for pattern in patterns:
                if re.search(pattern, text):
                    detected_text = text
                    detected_texts.append(detected_text)
                    break

    if len(detected_texts) == 0:
        return "Oops, no pattern match text found"

    # Assuming we need the first valid chassis number found
    CHASSIS_NO = detected_texts[0] if detected_texts else "Oops, no pattern match text found"
    return CHASSIS_NO

def process_ocr_result_engine_no(result):
    # Define regex patterns
    patterns = [
        r'^(?=.*\d.*\d.*\d.*\d)[A-Z0-9-]+$'  # text containing only numbers and capital letters, "-" is optional, at least 4 numbers in the text
    ]

    detected_texts = []
    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            for pattern in patterns:
                if re.search(pattern, text):
                    detected_text = text
                    detected_texts.append(detected_text)
                    break

    if len(detected_texts) == 0:
        return "Oops, no pattern match text found"

    # Assuming we need the first valid engine number found
    ENGINE_NO = detected_texts[0] if detected_texts else "Oops, no pattern match text found"
    return ENGINE_NO

def process_ocr_result_cylinder_capacity(result):
    regex_pattern = r'(CC|0000|00)'  # Matches 'CC', '0000', or '00'
    detected_texts = []

    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            if re.search(regex_pattern, text):
                detected_text = text
                detected_texts.append(detected_text)

    if len(detected_texts) == 0:
        return "Oops, No pattern match text found"

    modified_texts = []
    for text in detected_texts:
        modified_text = text.replace('A', '4').replace('Z', '7').replace('F', '8')

        if '.' in modified_text:
            modified_text = modified_text.split('.')[0] + '.00 CC'
        elif modified_text.endswith('0000'):
            modified_text = modified_text[:-4] + '.00 CC'
        else:
            modified_text += ' CC'  # Adding ' CC' if neither condition matches

        modified_texts.append(modified_text)

    # Return the first modified text found
    CYLINDER_CAPACITY = modified_texts[0] if modified_texts else "Oops, No pattern match text found"
    return CYLINDER_CAPACITY

def process_ocr_result_class_of_vehicle(result):
    # Define the target strings for fuzzy matching
    target_strings = ["MOTOR TRICYCLE", "MOTOR CYCLE", "LAND VEHICLE", "DUALPURPOSE VEHICLE"]

    text_list = []
    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            text_list.append(text)

    # Find target strings using fuzzy matching and update the text based on conditions
    for text in text_list:
        for target in target_strings:
            if fuzz.ratio(target, text) > 50:  # adjust the threshold as needed
                # Check if the text starts with "MOT" or the 4th and 5th letters are "OR"
                if text.startswith("MOT") or (len(text) > 4 and text[3:5] == "OR"):
                    text = "MOTOR" + text[5:]

                # Update text based on specified conditions
                if "DUAL" in text:
                    text = "DUAL PURPOSE VEHICLE"
                elif "LAND" in text:
                    text = "LAND VEHICLE"
                elif "TRI" in text:
                    text = "MOTOR TRICYCLE"

                # Ensure "MOTOR" is followed by a space if it is a single word
                if "MOTOR" in text and " " not in text.split("MOTOR", 1)[1]:
                    text = text.replace("MOTOR", "MOTOR ", 1)

                # Check if any word has 5 letters and ends with "CLE", replace with "CYCLE"
                words = text.split(" ")
                words = [word if not (len(word) == 5 and word.endswith("CLE")) else "CYCLE" for word in words]
                text = " ".join(words)

                return text  # Return the text once a match is found

    return "Oops, no pattern match text found"

def process_ocr_result_taxation_class(result):
    # Define the target strings for fuzzy matching
    target_strings = ["THREE WHEELER CAR", "MOTOR CYCLE", "LAND VEHICLE", "DUAL PURPOSE VEHICLE", "LIGHT MOTOR CYCLE", "MOTOR CAR"]

    text_list = []
    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)

            # Remove any single letter after a whitespace
            words = text.split()
            text = ' '.join([word for word in words if len(word) > 1])

            # If "MOTOR" is in the text, check the next character is a whitespace or not
            if "MOTOR" in text:
                motor_index = text.find("MOTOR")
                if motor_index != -1 and motor_index + 5 < len(text):
                    next_char = text[motor_index + 5]
                    if next_char != ' ':
                        text = text[:motor_index + 5] + ' ' + text[motor_index + 5:]

            text_list.append(text)

    # Find target strings using fuzzy matching and update the text based on conditions
    for text in text_list:
        for target in target_strings:
            if fuzz.ratio(target, text) > 50:  # adjust the threshold as needed

                # Update the text based on the specified conditions
                if "DUAL" in text:
                    text = "DUAL PURPOSE VEHICLE"
                elif "LAND" in text:
                    text = "LAND VEHICLE"
                elif "THREE" in text:
                    text = "THREE WHEELER CAR"

                return text  # Return the text once a match is found

    return "Oops, no pattern match text found"

def process_ocr_result_status_when_reg(result):
    # Define the target strings for fuzzy matching
    target_strings = ["BRAND NEW","RECONDITIONED"]

    text_list = []
    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            text_list.append(text)

    # Find target strings using fuzzy matching and update the text based on conditions
    for text in text_list:
        for target in target_strings:
            if fuzz.ratio(target, text) > 50:  # adjust the threshold as needed
                #print("STATUS WHEN REGISTERED:", target)
                return target  # Return the target string once a match is found

    return "Oops, no pattern match text found"

def process_ocr_result_fuel_type(result):
    # Define the target strings for fuzzy matching
    target_strings = ["DIESEL","PETROL"]

    text_list = []
    for page in result:
        if page is None:
            continue
        for line in page:
            if line is None:
                continue
            text = line[1][0]
            text = str(text)
            text_list.append(text)

    # Find target strings using fuzzy matching and update the text based on conditions
    for text in text_list:
        for target in target_strings:
            if fuzz.ratio(target, text) > 50:  # adjust the threshold as needed
                #print("FUEL TYPE:", target)
                return target  # Return the target string once a match is found

    return "Oops, no pattern match text found"



# Define a dictionary to store OCR coordinates and thresholds for each field
field_params = {
    "REGISTRATION_NO": ([9, 14, 12, 10], 12.5, process_ocr_result_reg_no),
    "CHASSIS_NO": ([55, 12, 69, 10], 11, process_ocr_result_chassis_no),
    "ENGINE_NO": ([9, 49, 13, 45], 10, process_ocr_result_engine_no),
    "CYLINDER_CAPACITY": ([56, 49, 61, 45], 6, process_ocr_result_cylinder_capacity),
    "CLASS_OF_VEHICLE": ([10, 52, 16, 47], 8, process_ocr_result_class_of_vehicle),
    "TAXATION_CLASS": ([56, 52, 64, 48], 10, process_ocr_result_taxation_class),
    "STATUS_WHEN_REGISTERED": ([10, 55, 14, 50], 7, process_ocr_result_status_when_reg),
    "FUEL_TYPE": ([56, 55, 58, 50], 7, process_ocr_result_fuel_type)
}

# Combined pipeline function to process all fields for a given image
def combined_pipeline_all_fields(image_path, field_params):
    results = {}
    cropped_image = outline_detection_pipeline(image_path)
    
    for field, params in field_params.items():
        ocr_coords_percentage, threshold_percentage, process_ocr_result_func = params
        result = perform_ocr_on_cropped_image(cropped_image, ocr_coords_percentage, threshold_percentage)
        results[field] = process_ocr_result_func(result)
    
    return results

# Streamlit interface
st.title("Vehicle Registration Information Extractor 🚗📋")

col1, col2 = st.columns([2, 3])

with col1:
    st.write("Upload an image of a vehicle CR book:")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Process Image'):
            with st.spinner("Processing image..."):
                # Convert PIL image to OpenCV format
                image_np = np.array(image.convert('RGB'))
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                image_path = 'temp_image.jpg'
                cv2.imwrite(image_path, image_cv)
                
                # Process the image
                results = combined_pipeline_all_fields(image_path, field_params)
                
                # Store the results in session state
                for field, value in results.items():
                    st.session_state[field] = value

with col2:
    st.write("Extracted Details")
    form = st.form(key='extracted_details_form')
    labels = [
        "REGISTRATION_NO", "CHASSIS_NO", "ENGINE_NO", "CYLINDER_CAPACITY", "CLASS_OF_VEHICLE",
        "TAXATION_CLASS", "STATUS_WHEN_REGISTERED", "FUEL_TYPE"
    ]

    # Use session state to store and update form values
    for label in labels:
        default_value = st.session_state.get(label, '')
        st.session_state[label] = form.text_input(label.replace("_", " ").title(), default_value)

    form.form_submit_button('Submit')