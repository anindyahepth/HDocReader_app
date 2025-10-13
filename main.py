from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
import sqlite3
import base64
import json
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv
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
import time
from mlflow_config import setup_mlflow, log_prediction_run




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

import subprocess

#manual ngrok set up for Colab
from pyngrok import ngrok #needed for running on Colab
#################################
ngrok_path = ngrok.install_ngrok()
FLASK_PORT = 5000
public_url = ngrok.connect(FLASK_PORT).public_url
print(f"ngrok tunnel established! Public URL: {public_url}")
#################################


# Flask app configuration for local deployment
load_dotenv()
FLASK_PORT = os.environ.get('FLASK_APP_PORT')
MLFLOW_PORT = os.environ.get('MLFLOW_UI_PORT')
DASHBOARD_PORT = os.environ.get('DASHBOARD_PORT')

app = Flask(__name__)
app.secret_key = os.urandom(24)

auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("your_password")  # Replace with a strong password
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

# Setup MLflow
mlflow = setup_mlflow()
def start_mlflow_ui(port):
    print(f"Starting MLflow UI on port {port}...")
    try:
        # Get the absolute path to ensure MLflow UI uses the correct location
        current_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Change to the app directory before starting MLflow UI
        original_dir = os.getcwd()
        os.chdir(current_dir)
        
        # Start MLflow UI on specified port with relative paths from app directory
        mlflow_process = subprocess.Popen([
            "mlflow", "ui", 
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
            "--host", "0.0.0.0",
            "--port", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=current_dir)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        print(f"   Database: {os.path.join(current_dir, 'mlflow.db')}")
        print(f"   Artifacts: {os.path.join(current_dir, 'mlruns')}")
        return mlflow_process
    except Exception as e:
        print(f"Failed to start MLflow UI: {e}")
        return None

 # Start Dashboard
def start_dashboard(port):
    print(f"Starting Dashboard on port {port}...")
    try:
        dashboard_process = subprocess.Popen([
            sys.executable, "real_time_dashboard.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return dashboard_process
    except Exception as e:
        print(f" Failed to start Dashboard: {e}")
        return None

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    gemini_model = None


@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       data = request.form['drawing_data'] 
       encoded = data.split(',')[1]      
       image_data = base64.b64decode(encoded)
       print("Base64 decoded successfully. Length:", len(image_data))
       
       #save as png file
       output_dir = './images'
       os.makedirs(output_dir, exist_ok=True)
       image_filename = os.path.join(output_dir, f"image_01.png") 
       with open(image_filename, "wb") as f: 
              f.write(image_data)  
              
       jpg_filename = os.path.join(output_dir, f"image_01.jpg")
       utils.convert_png_to_jpg_pillow_alpha_fill(image_data, jpg_filename)
       
       
       # Start timing
       start_time = time.time()
       
       predicted_text_list = utils.make_predictions(jpg_filename)
       
       predicted_text = utils.format_predicted_text(predicted_text_list)

       draft_transcript = predicted_text

       image_path = jpg_filename

       final_transcript = utils.correct_transcript_with_gemini(gemini_model, draft_transcript, image_path)
       
       # Calculate processing time
       processing_time = time.time() - start_time
       
       # Initialize CER evaluator with GPT-4o for reference transcription
       from accuracy_evaluator import AccuracyEvaluator
       
       # Use OpenAI for reference transcription
       openai_api_key = os.environ.get("OPENAI_API_KEY")
       if openai_api_key:
           evaluator = AccuracyEvaluator(
               api_key=openai_api_key,
               provider="openai",
               model_name="gpt-4o"
           )
       else:
           # Fallback to basic CER if OpenAI not available
           evaluator = AccuracyEvaluator(
               api_key="",
               provider="cer",
               model_name="character_error_rate"
           )
       
       # Evaluate prediction quality using CER
       evaluation_result = evaluator.evaluate_prediction(
           image_path=image_path,
           prediction=final_transcript,
           reference_text= None  
       )
       
       #Log to MLflow with metrics and evaluation results per run
       log_prediction_run(
           draft_text=draft_transcript,
           corrected_text=final_transcript,
           image_path=image_path,
           processing_time=processing_time,
           evaluation_result=evaluation_result
       )           

       print("TrOCR_draft:",predicted_text)
       print("final_transcript:",final_transcript)
       print("GPT-4o Reference:", evaluation_result.get("reference_text", "N/A"))
       print("CER:", evaluation_result.get("cer", "N/A"))
       print("Character Accuracy:", evaluation_result.get("character_accuracy", "N/A"))
       print(processing_time)
       print("Flask_app_port:",FLASK_PORT)       
       
    
       conn = sqlite3.connect('db.db')
       c = conn.cursor()
       c.execute("INSERT INTO drawings (data, predicted_text) VALUES (?, ?)", (data, final_transcript))
       conn.commit()
       conn.close()
       return jsonify({'prediction': final_transcript})
   return render_template("index.html")

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
      start_mlflow_ui(port=MLFLOW_PORT)
      time.sleep(3)
      print(f"MLFLOW UI STARTED ON http://0.0.0.0:{MLFLOW_PORT}")
      
      start_dashboard(port=DASHBOARD_PORT)
      time.sleep(3)
      print(f"DASHBOARD STARTED ON http://0.0.0.0:{DASHBOARD_PORT}")
      
      # Start Flask app last (blocking)
      print(f"FLASK APP STARTING ON http://0.0.0.0:{FLASK_PORT}")
      app.run(host='0.0.0.0', port=FLASK_PORT)
      
