import os
import pytesseract
from PIL import Image

img_dir = 'TestMe'

word = input("Enter the word to search for: ")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith('.png') or img_file.lower().endswith('.jpg') or img_file.lower().endswith('.jpeg'):
            img_path = os.path.join(img_dir, img_file)
            with Image.open(img_path) as img:
                text = pytesseract.image_to_string(img).lower()
            if word.lower() in text:
                print(f"{img_file} contains the word '{word}'")
except Exception as e:
    print(f"An error occurred: {str(e)}")
