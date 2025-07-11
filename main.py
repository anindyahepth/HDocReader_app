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
from utils import utils




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

#manual ngrok set up for Colab
#################################
ngrok_path = ngrok.install_ngrok()
FLASK_PORT = 5000
public_url = ngrok.connect(FLASK_PORT).public_url
print(f"ngrok tunnel established! Public URL: {public_url}")
#################################

app = Flask(__name__)
app.secret_key = os.urandom(24)

auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("passwd")  # Replace with a strong password
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


# --- Configure Gemini API ---
os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    # print("Please ensure your GOOGLE_API_KEY is set correctly and you have access to gemini-pro-vision.")
    gemini_model = None # Set to None to handle errors downstream


  
  
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
       
       utils.convert_png_to_jpg_pillow_alpha_fill(image_data, jpg_filename)
       
       
       #open the jpg file, grey-scale convert and feed it to the model
              
       #image = Image.open(jpg_filename).convert('L')
       
       predicted_text_list = utils.make_predictions(jpg_filename)
       
       predicted_text = utils.format_predicted_text(predicted_text_list)

       draft_transcript = predicted_text

       image_path = jpg_filename

       final_transcript = utils.correct_transcript_with_gemini(gemini_model, draft_transcript, image_path)
       
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
