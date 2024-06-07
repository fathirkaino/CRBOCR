import streamlit as st
from PIL import Image
import numpy as np
from passporteye import read_mrz
from paddleocr import PaddleOCR
import re
from fuzzywuzzy import fuzz
import pandas as pd

# Setup the page
st.set_page_config(page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Load the countries data from CSV to create a mapping of ISO codes to country names
country_df = pd.read_csv('countries.csv')
country_dict = dict(zip(country_df['ISO Code'], country_df['Country']))

# Display the application title in a creative format
st.markdown("""
#  🤖 **AI-Powered Document Info Extractor**
""")

st.write(" 🔍 One-stop solution for extracting information from various documents efficiently. Navigate through the tabs to start processing your documents !")

# Initialize OCR
ocr = PaddleOCR(lang='en', use_gpu=False)

# Function to extract MRZ data from the uploaded image using PassportEye
def extract_mrz_data(image):
    mrz = read_mrz(image, save_roi=True)
    if mrz is None:
        return {}
    mrz_data = mrz.to_dict()
    # Rename keys and clean up data for clarity
    mrz_data['nic_number'] = mrz_data.pop('personal_number', '').replace('<', '')
    mrz_data['passport_number'] = mrz_data.pop('number', '').replace('<', '')
    mrz_data['mrz_code'] = mrz_data.pop('raw_text', '')
    mrz_data['sex'] = 'Female' if mrz_data.get('sex', '') == 'F' else 'Male'
    mrz_data['country_code'] = mrz_data.pop('nationality', '').replace('<', '')
    mrz_data['country'] = country_dict.get(mrz_data['country_code'], '')  # Get country name from country code

    # Format the expiration date and calculate issue date
    expiration = mrz_data.get('expiration_date', '')
    if len(expiration) == 6:
        year, month, day = expiration[:2], expiration[2:4], expiration[4:]
        mrz_data['expiration_date'] = f"20{year}-{month}-{day}"
        issue_year = str(int("20" + year) - 10)
        mrz_data['issue_date'] = f"{issue_year}-{month}-{day}"
    else:
        mrz_data['issue_date'] = ''

    # Format the date_of_birth and determine year prefix
    dob = mrz_data.get('date_of_birth', '')
    if len(dob) == 6:
        year, month, day = dob[:2], dob[2:4], dob[4:]
        year_prefix = "19" if "V" in mrz_data['nic_number'] else "20"
        mrz_data['date_of_birth'] = f"{year_prefix}{year}-{month}-{day}"
    else:
        mrz_data['date_of_birth'] = ''

    return mrz_data

# Function to extract text from image using PaddleOCR
def extract_text_from_image(image):
    image_array = np.array(image)
    result = ocr.ocr(image_array, rec=True)
    return " ".join([line[1][0] for line in result[0]])


def process_ocr_results(ocr_results):
    extracted_info = {
        "Driving Licence No": None,
        "National Identification Card No": None,
        "Name": None,
        #"Address": [],
        "Address": None,
        "Data Of Birth": None,
        "Date Of Issue": None,
        "Date Of Expiry": None,
        "Blood Group": None
    }

    for page in ocr_results:
        for i, line in enumerate(page):
            text = line[1][0]
            if re.search(r'^5\.(B|8)\d+', text):
                match = re.sub(r'^5\.', '', text)
                extracted_info["Driving Licence No"] = "B" + match[1:]
            
            elif re.search(r'^B\d{7}', text):
                extracted_info["Driving Licence No"] = text
            

            elif re.search(r'\d{9,}', text):
                match = re.search(r'\d{9,}[A-Za-z]*', text)
                if match:
                    extracted_info["National Identification Card No"] = match.group()
                else:
                    extracted_info["National Identification Card No"] = text


            elif re.search(r"^(1,2\.|\.2|1\.2\.|12\.|1,2,|1\.2,|,2).+$", text):
                match = re.sub(r'\d+', '', text)  # Remove numbers from the text
                match = re.sub(r'[,.]', '', match)  # Remove commas and periods from the text
                # Check if there's another line to process
                if i + 1 < len(page):
                    temp = page[i + 1][1][0]
                    if temp == "SL":
                        temp = ""
                    if re.search(r'^(8|B)\.', temp):
                        temp = ""
                else:
                    temp = ""  # No further line to process
                merge_name = f"{match} {temp}".strip()  # Merge and strip any extra spaces
                extracted_info["Name"] = merge_name
                

            elif re.search(r'^(8|B)\.', text):
                match = text[2:]  # Remove prefix and capture the rest of the text
                temp_list = [page[j][1][0] for j in range(i+1, min(i+3, len(page)))] if i + 2 < len(page) else [page[j][1][0] for j in range(i+1, len(page))] if i + 1 < len(page) else []

                # Remove 'SL' from temp_list
                temp_list = [line for line in temp_list if 'SL' not in line]

                # Remove any strings that match the date pattern
                temp_list = [line for line in temp_list if not re.match(r'^(3|5)\.\d{2}\.\d{2}\.\d{4}', line)]

                # Merge 'match' with the remaining values in temp_list
                merge = ' '.join([match] + temp_list)

                extracted_info["Address"] = merge

            elif re.search(r'^(3|5)\.\d{2}\.\d{2}\.\d{4}', text):
                extracted_info["Data Of Birth"] = text.split('.', 1)[1].strip()

            elif re.search(r'^4(a|s)\.\d{2}\.\d{2}\.\d{4}', text):
                extracted_info["Date Of Issue"] = text.split('.', 1)[1].strip()

            elif re.search(r'^4(b|6)\.\d{2}\.\d{2}\.\d{4}', text):
                extracted_info["Date Of Expiry"] = text.split('.', 1)[1]. strip()

            #elif re.search(r'^Blood', text, re.IGNORECASE):
            #    extracted_info["Blood Group"] = text.split(None, 2)[-1]

            elif re.search(r'^Blood', text, re.IGNORECASE):
                match = text  # Current line
                # Check if there's another line to process
                if i + 1 < len(page):
                    temp = page[i + 1][1][0]
                    # Check if the next line contains '+'
                    if '+' not in temp:
                        temp = ""
                else:
                    temp = ""  # No further line to process

                # Merge 'match' with 'temp' if 'temp' is not empty
                merge = f"{match} {temp}".strip() if temp else match.strip()
                # Extract the blood group information
                extracted_info["Blood Group"] = merge.split(None, 2)[-1]

        return extracted_info


def load_and_process_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    result = ocr.ocr(image_np, rec=True)

    # Collect all OCR text for display
    ocr_texts = []
    for page in result:
        for line in page:
            text = line[1][0]
            print(text)
            ocr_texts.append(text)
    #print(ocr_texts)

    return image, process_ocr_results(result), ocr_texts


# Function to extract vehicle details using OCR
def extract_key_value(ocr_results, key_name, line_param, value_index, fuzz_score_threshold=80, threshold=10):
    mid_height_results = []
    for coordinates, (text, _) in ocr_results:
        mid_height = (coordinates[0][1] + coordinates[3][1]) / 2
        mid_height_results.append(((coordinates[0], coordinates[3]), (text, _), mid_height))
    sorted_results = sorted(mid_height_results, key=lambda x: x[2])
    key_match = None
    for (_, _), (text, _), mid_height in sorted_results:
        if fuzz.partial_ratio(key_name.replace(" ", ""), text) >= fuzz_score_threshold:
            key_match = text
            break

    if key_match is None:
        return None

    key_mid_height = None
    for (_, _), (text, _), mid_height in sorted_results:
        if text == key_match:
            key_mid_height = mid_height
            break

    if key_mid_height is None:
        return None

    values = []
    for (_, _), (text, _), mid_height in sorted_results:
        if line_param == 'same_line' and abs(mid_height - key_mid_height) <= threshold:
            values.append(text)
        elif line_param == 'next_line' and mid_height > key_mid_height + threshold:
            values.append(text)

    # Handling value_index which can be an int or a list
    if isinstance(value_index, list):
        return [values[i] for i in value_index if 0 <= i < len(values)]
    elif 0 <= value_index < len(values):
        return values[value_index]
    else:
        return None


def extract_details_from_image(image):
    image_array = np.array(image)  # Convert to NumPy array
    result = ocr.ocr(image_array, rec=True)
    ocr_results = result[0] 

    # Define key-value pairs with fuzz_score_threshold and threshold
    key_value_pairs = [
        ("Registration No.", "next_line", 0, 80, 30),
        ("Chassis No.", "next_line", 1, 80, 30),
        ("Current Owner/Address/ID.No.", "next_line", [0, 1], 80, 60),
        ("Conditions/Special Notes", "next_line", 0, 80, 10),
        ("Absolute Owner", "next_line", [0, 1, 2], 80, 20),
        ("Engine No", "next_line", 0, 80, 20),
        ("Cylinder Capacity (cc)", "next_line", 1, 80, 20),
        ("Class of Vehicle", "next_line", 0, 80, 20),
        ("Taxation Class", "next_line", 1, 80, 20),
        ("Status when Registered", "next_line", 0, 80, 10),
        ("Make", "next_line", 0, 80, 10),
        ("Model", "next_line", 1, 80, 20),
        ("Wheel Base", "next_line", 0, 80, 10),
        ("Type of Body", "next_line", 0, 80, 20)
    ]

    extracted_details = {}

    for key_name, value_in, value_at, fuzz_score_threshold, threshold in key_value_pairs:
        result = extract_key_value(ocr_results, key_name, value_in, value_at, fuzz_score_threshold, threshold)
        extracted_details[key_name] = result

    return extracted_details

# Streamlit tab setup
tab1, tab2, tab3 = st.tabs(["🪪 Driving License Information Extractor", "🛂 Passport Information Extractor", "🚗 Vehicle CR Book Information Extractor"])

with tab1:
    st.title('🪪 Driving License Information Extractor')

    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("Upload an image of a driving license:")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key='license_uploader')
        if uploaded_image is not None:
            image, extracted_info, ocr_texts = load_and_process_image(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Process Image', key='process_license_image'):
                st.session_state["extracted_info"] = extracted_info
                st.session_state["ocr_texts"] = ocr_texts  # Store OCR texts in session state
                if ocr_texts:
                    with st.expander("View OCR Extracted Texts"):
                        for text in ocr_texts:
                            st.text(text)  # Display each line separately

    with col2:
        st.write("Extracted Details")
        form = st.form(key='license_details_form')
        labels = [
            "Driving Licence No", "National Identification Card No", "Name", "Address", 
            "Data Of Birth", "Date Of Issue", "Date Of Expiry", "Blood Group"
        ]
        for label in labels:
            default_value = st.session_state.get("extracted_info", {}).get(label, '')
            if isinstance(default_value, list):
                default_value = ', '.join(default_value)
            form.text_input(label, default_value)
        form.form_submit_button('Submit')

with tab2:
    st.title("🛂 Passport Information Extractor ")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("Upload an image of the passport:")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key='passport_uploader')
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Process Image', key='process_passport_image'):
                with st.spinner("Extracting data from passport..."):
                    mrz_data = extract_mrz_data(uploaded_image)
                    extracted_text = extract_text_from_image(image)
                    if "passport" in extracted_text.lower():
                        st.success("🔍 Passport found! 🎉")
                    for label, value in mrz_data.items():
                        st.session_state[label] = value

    with col2:
        st.write("Extracted Passport Details")
        form = st.form(key='passport_details_form')
        labels = ["names", "surname", "country_code", "country", "nic_number", "passport_number", "date_of_birth", "expiration_date", "issue_date", "sex", "type", "mrz_code"]

        for label in labels:
            default_value = st.session_state.get(label, '')
            # Customize display names where needed
            display_label = {
                "country_code": "Country Code",
                "country": "Country",  # New field for full country name
                "issue_date": "Issue Date"  # New field for issue date
            }.get(label, label.capitalize())
            st.session_state[label] = form.text_input(display_label, default_value)

        form.form_submit_button('Submit')


with tab3:
    st.title("🚗 Vehicle CR Book Information Extractor ")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("Upload an image of a vehicle CR book:")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key='vehicle_uploader')
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Process Image', key='process_vehicle_image'):
                with st.spinner("Processing image..."):
                    extracted_details = extract_details_from_image(image)
                    for label, value in extracted_details.items():
                        st.session_state[label] = value if value else ''

    with col2:
        st.write("Extracted Details")
        form = st.form(key='vehicle_details_form')
        labels = [
            "Registration No.", "Chassis No.", "Current Owner/Address/ID.No.", "Conditions/Special Notes",
            "Absolute Owner", "Engine No", "Cylinder Capacity (cc)", "Class of Vehicle",
            "Taxation Class", "Status when Registered", "Make", "Model", "Wheel Base", "Type of Body"
        ]
        for label in labels:
            default_value = st.session_state.get(label, '')
            form.text_input(label, default_value)
        form.form_submit_button('Submit')
