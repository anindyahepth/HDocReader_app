from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from pyngrok import ngrok #needed for Colab
import sqlite3
import base64
import json
import numpy as np
from PIL import Image
import io
import os
from io import BytesIO
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
import torch.utils.data
import zipfile

import re
import sys
import argparse
import ast
from collections import OrderedDict
import torchvision.transforms.functional as TF
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import google.generativeai as genai


import cv2




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

#manual ngrok set up for Colab
#################################
ngrok_path = ngrok.install_ngrok()
FLASK_PORT = 5000
public_url = ngrok.connect(FLASK_PORT).public_url
print(f"ngrok tunnel established! Public URL: {public_url}") # this is a crucial step on Colab
#################################

app = Flask(__name__)
app.secret_key = os.urandom(24)

auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("adey")  # Replace with a strong password
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('db.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drawings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB, predicted_text TEXT)''')
    conn.commit()
    conn.close()

init_db()
 
 
 
         
def convert_png_to_jpg_pillow_alpha_fill(png_data, jpg_filename="output.jpg"):
    try:
        img = Image.open(io.BytesIO(png_data))
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))  # White background
            background.paste(img, mask=img.split()[3])  # Paste with alpha mask
            img = background
        img.save(jpg_filename, "JPEG")
        print(f"PNG converted to JPG (Pillow alpha fill) and saved as {jpg_filename}")
    except Exception as e:
        print(f"Error (Pillow alpha fill): {e}")
        
        
        
        
def split_handwritten_page(image_path, output_dir="lines", target_size=(512, 64)):

    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Preprocessing: Binarization
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal Projection
    horizontal_projection = np.sum(binary_img, axis=1)

    # Find line boundaries
    line_starts = []
    line_ends = []
    threshold = np.max(horizontal_projection) / 120  # Adjust threshold as needed - 100 works well
    in_line = False

    for y, projection_value in enumerate(horizontal_projection):
        if projection_value > threshold and not in_line:
            line_starts.append(y)
            in_line = True
        elif projection_value <= threshold and in_line:
            line_ends.append(y)
            in_line = False

    # Handle the case where the last line extends to the bottom
    if in_line:
        line_ends.append(binary_img.shape[0])

    line_images = []

    for i, (start_y, end_y) in enumerate(zip(line_starts, line_ends)):
        line_img = img[start_y:end_y, :]

        # Pad or resize to target size
        pil_img = Image.fromarray(line_img)
        resized_img = pil_img
        #resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        line_images.append(resized_img)

        # Save line image
        line_filename = os.path.join(output_dir, f"line_{i}.jpg")
        resized_img.save(line_filename)

    return line_images    

        


def recognize_text(image, processor):
    """
    Recognizes text in an image using the TrOCR model.

    Args:
        image_bytes (bytes): The bytes of the JPEG image file.

    Returns:
        str: The recognized text.
    """
    try:
        
        image = image  # No need to convert, already RGB

        # Convert image to RGB if it's grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = pixel_values.to(device)  # Move to the correct device

        # Generate predictions
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        # Decode the predicted IDs into text
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return predicted_text
    except Exception as e:
        print(f"Error during text recognition: {e}")
        return ""  # Return empty string on error




def make_predictions(image_path):

  # Load the TrOCR model and processor
  processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
  model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

  # Set the model to evaluation mode
  model.eval()

  # Use GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  
  line_images = split_handwritten_page(image_path) #line tensors
  preds_list = []

  for i,line_image in enumerate(line_images):
         image = line_image

         with torch.no_grad():
          predicted_text = recognize_text(image, processor = processor) 
         preds_list.append(predicted_text)
        
  return preds_list




def format_predicted_text(predicted_text_list):
    """
    Formats a list of strings (or nested lists of strings) into a single string,
    with individual strings appearing on different lines.

    Args:
        predicted_text_list (list): A list of strings or nested lists of strings.

    Returns:
        str: A formatted string with newlines.
    """
    formatted_text = ""
    for item in predicted_text_list:
        if isinstance(item, list):
            formatted_text += "\n".join(item) + "\n"  # Join inner list with newlines, add extra newline
        else:
            formatted_text += str(item) + "\n"  # Add newline after each item

    return formatted_text  # Remove leading/trailing newlines


# --- Configure Gemini API ---
os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Please ensure your GOOGLE_API_KEY is set correctly and you have access to gemini-pro-vision.")
    gemini_model = None # Set to None to handle errors downstream

def correct_transcript_with_gemini(draft_transcript: str, image_path: str) -> str:
    
    if gemini_model is None:
        return "Error: Gemini model not initialized. Check API key and model access."

    try:

        image = Image.open(image_path) 
        

        # Prepare the prompt for Gemini
        prompt_parts = [
            "You are an expert transcriber specializing in historical handwritten documents and accurate optical character recognition (OCR).",
            "Review the following draft transcript of a single line of handwritten text.",
            "Using the provided image as the authoritative source, meticulously correct any errors, omissions, or misinterpretations in the draft.",
            "Pay extremely close attention to spelling, punctuation, capitalization, and spacing exactly as it appears in the handwritten image.",
            "If the draft is entirely incorrect or misses major parts, provide the full correct transcription based on the image.",
            "If the draft is mostly correct, make only the necessary minor corrections.",
            "Do NOT add any explanations or additional text; only provide the corrected transcript.",
            "\n\n**Draft Transcript:**\n",
            f"{draft_transcript}\n\n",
            "**Image Context:**\n",
            image, # Gemini takes the PIL Image object directly
            "\n\n**Corrected Transcript:**\n"
        ]

        # Call the Gemini API
        response = gemini_model.generate_content(prompt_parts)

        # Extract the corrected text
        corrected_transcript = response.text.strip()

        if not corrected_transcript:
            return "Gemini returned an empty correction."

        return corrected_transcript

    except Exception as e:
        print(f"Error during transcript correction: {e}")
        # In a production Flask app, you might log this error more formally
        return f"An error occurred during correction: {e}"

  
  
@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       data = request.form['drawing_data'] 
       encoded = data.split(',')[1]      
       image_data = base64.b64decode(encoded)
       print("Base64 decoded successfully. Length:", len(image_data))
       
       #save as png file 
             
       #output_dir = '/Users/anindyadey/HTR-app/images'
       output_dir = './images'
       image_filename = os.path.join(output_dir, f"image_01.png") 
       with open(image_filename, "wb") as f: 
              f.write(image_data)
              
       #convert png to jpg file and save
              
       jpg_filename = os.path.join(output_dir, f"image_01.jpg")
       
       convert_png_to_jpg_pillow_alpha_fill(image_data, jpg_filename)
       
       
       #open the jpg file, grey-scale convert and feed it to the model
              
       #image = Image.open(jpg_filename).convert('L')
       
       predicted_text_list = make_predictions(jpg_filename)
       
       predicted_text = format_predicted_text(predicted_text_list)

       draft_transcript = predicted_text

       image_path = jpg_filename

       final_transcript = correct_transcript_with_gemini(draft_transcript, image_path)
       
       print(predicted_text)
       print(final_transcript)
       
       
        

       conn = sqlite3.connect('db.db')
       c = conn.cursor()
       c.execute("INSERT INTO drawings (data, predicted_text) VALUES (?, ?)", (data, final_transcript))
       conn.commit()
       conn.close()
       return jsonify({'prediction': final_transcript})
   return render_template('index.html')

#sqlite3.Binary(data.encode('utf-8'))
#print(type(data))
#print(data)
#data = data.encode('utf-8').decode('utf-8') #add explicit encoding.



@app.route('/admin')
@auth.login_required
def admin():
    conn = sqlite3.connect('db.db')
    c = conn.cursor()
    c.execute("SELECT id, data FROM drawings")
    drawings = c.fetchall()
    conn.close()
    return render_template('admin.html', drawings=drawings)

@app.route('/export')
@auth.login_required
def export():
    conn = sqlite3.connect('db.db')
    c = conn.cursor()
    c.execute("SELECT id, data FROM drawings")
    drawings = c.fetchall()
    conn.close()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for id, data in drawings:
            img_data = base64.b64decode(data.split(',')[1])
            zip_file.writestr(f'drawing_{id}.png', img_data)

    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name='drawings.zip', mimetype='application/zip')

@app.route('/delete/<int:drawing_id>')
@auth.login_required
def delete(drawing_id):
    conn = sqlite3.connect('db.db')
    c = conn.cursor()
    c.execute("DELETE FROM drawings WHERE id = ?", (drawing_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin'))

@app.route('/delete_all')
@auth.login_required
def delete_all():
    conn = sqlite3.connect('db.db')
    c = conn.cursor()
    c.execute("DELETE FROM drawings")
    conn.commit()
    conn.close()
    return redirect(url_for('admin'))

if __name__ == '__main__':

      #app.run(host='0.0.0.0', port=8080)
      app.run(port=FLASK_PORT)
