from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
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

import imageio.v3 as iio




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from model.HTR_VT import MaskedAutoencoderViT
from functools import partial

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

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
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB, predicted_text TEXT, confidence TEXT)''')
    conn.commit()
    conn.close()

init_db()

    
 

def dict_from_file_to_list(filepath):
    try:
        with open(filepath, 'r') as file:
            dict_str = file.read()
            dictionary = ast.literal_eval(dict_str)
            return dictionary

    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return None
    except ValueError as e:
        print(f"Error parsing the file: {e}")
        return None
        
        
        

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
        
        
def inspect_png_pixel_values(png_data):
    try:
        img = Image.open(io.BytesIO(png_data))
        img_array = np.array(img)
        print(f"Image shape: {img_array.shape}")
        print(f"Min pixel value: {np.min(img_array)}")
        print(f"Max pixel value: {np.max(img_array)}")
    except Exception as e:
        print(f"Error inspecting pixel values: {e}")

def inspect_png_pixel_values(png_data):
    try:
        img = Image.open(io.BytesIO(png_data))
        img_array = np.array(img)
        print(f"Image shape: {img_array.shape}")
        print(f"Min pixel value: {np.min(img_array)}")
        print(f"Max pixel value: {np.max(img_array)}")
    except Exception as e:
        print(f"Error inspecting pixel values: {e}")

        
               
def transform_image(image):

    transformation = transforms.Compose([
        transforms.Resize(tuple([64, 512])),
        transforms.ToTensor(),
        
    ])
   # image_data = image.split(",", 1)
   # image_data = base64.b64decode(image_data)
   # image = Image.open(io.BytesIO(image_data)).convert('L')
    processed_image = transformation(image)
    #image_tensor = torch.reshape(processed_image[3], [1, 1, 64, 512])
    image_tensor = processed_image.unsqueeze(0)

    return image_tensor



def create_model_vitmae(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls=90,
                                 img_size=img_size,
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model



def make_predictions(image):
    
    
    pth_path = '/Users/anindyadey/HTR-app/best_CER.pth'
    dict_path = '/Users/anindyadey/HTR-app/dict_alph'

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model_vitmae(nb_cls =90, img_size= [64, 512])
    ckpt = torch.load(pth_path, map_location='cpu', weights_only = True)

    model_dict = OrderedDict()
    if 'model' in ckpt:
        ckpt = ckpt['model']

    unexpected_keys = ['state_dict_ema', 'optimizer']
    for key in unexpected_keys:
        if key in ckpt:
            del ckpt[key]


    model.load_state_dict(ckpt, strict= False)
    #model = model.to(device)
    model.eval()

    
    alpha = dict_from_file_to_list(dict_path)
    converter = utils.CTCLabelConverter(alpha)

    image_tensor = transform_image(image)
    print(image_tensor.shape)
    #image_tensor = torch.reshape(image_tensor[3], [1, 1, 64, 512]) #change here
    #print(image_tensor.shape)
    #image_tensor = image_tensor.to(device)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        predicted_text = preds_str[0]
        confidence = 1.
        
        return predicted_text, confidence, preds

  
  
@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       data = request.form['drawing_data'] 
       encoded = data.split(',')[1]      
       #header, encoded = data.split(",", 1)
       image_data = base64.b64decode(encoded)
       print("Base64 decoded successfully. Length:", len(image_data))
       #pil_image =  Image.open(io.BytesIO(image_data)).convert('L')
       #pil_image = Image.open(io.BytesIO(image_data)).convert('L') #change here
       
       output_dir = '/Users/anindyadey/HTR-app/images'
       image_filename = os.path.join(output_dir, f"image_01.png") 
       #pil_image.save(image_filename)
       with open(image_filename, "wb") as f: 
              f.write(image_data)
              
       #inspect_png_pixel_values(image_data)
       
       #print(type(pil_image))
  
       #image_path =  '/Users/anindyadey/HTR-app/images/drawing_20.jpg'   #change here
       #image = Image.open(image_filename).convert('L')
       
       jpg_filename = os.path.join(output_dir, f"image_01.jpg")
       
       convert_png_to_jpg_pillow_alpha_fill(image_data, jpg_filename)
       
       
       #image = Image.open(io.BytesIO(image_data))
       
       #if image.mode == 'RGBA':
       #     image = image.convert('RGB')
       #image.save(jpeg_filename, "JPEG")
       
       image = Image.open(jpg_filename).convert('L')
       
       predicted_text, confidence, preds = make_predictions(image)
       print(preds)
      

       conn = sqlite3.connect('db.db')
       c = conn.cursor()
       c.execute("INSERT INTO drawings (data, predicted_text, confidence) VALUES (?, ?, ?)", (data, predicted_text, confidence))
       conn.commit()
       conn.close()
       return jsonify({'prediction': predicted_text, 'confidence': float(confidence)})
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
    app.run(host='0.0.0.0', port=8080)



# from flask import Flask, render_template, request, send_file, redirect, url_for, session
# import sqlite3
# import base64
# import zipfile
# import io
# import os
# from flask_httpauth import HTTPBasicAuth
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)
# app.secret_key = os.urandom(24)

# auth = HTTPBasicAuth()

# users = {
#     "admin": generate_password_hash("adey")  # Replace with a strong password
# }

# @auth.verify_password
# def verify_password(username, password):
#     if username in users and check_password_hash(users.get(username), password):
#         return username

# # Initialize SQLite database
# def init_db():
#     conn = sqlite3.connect('db.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS drawings
#                  (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT)''')
#     conn.commit()
#     conn.close()

# init_db()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         data = request.form['drawing_data']
#         conn = sqlite3.connect('db.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO drawings (data) VALUES (?)", (data,))
#         conn.commit()
#         conn.close()
#         return redirect(url_for('index'))
#     return render_template('index.html')

# @app.route('/admin')
# @auth.login_required
# def admin():
#     conn = sqlite3.connect('db.db')
#     c = conn.cursor()
#     c.execute("SELECT id, data FROM drawings")
#     drawings = c.fetchall()
#     conn.close()
#     return render_template('admin.html', drawings=drawings)

# @app.route('/export')
# @auth.login_required
# def export():
#     conn = sqlite3.connect('db.db')
#     c = conn.cursor()
#     c.execute("SELECT id, data FROM drawings")
#     drawings = c.fetchall()
#     conn.close()

#     zip_buffer = io.BytesIO()
#     with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
#         for id, data in drawings:
#             img_data = base64.b64decode(data.split(',')[1])
#             zip_file.writestr(f'drawing_{id}.png', img_data)

#     zip_buffer.seek(0)
#     return send_file(zip_buffer, as_attachment=True, download_name='drawings.zip', mimetype='application/zip')

# @app.route('/delete/<int:drawing_id>')
# @auth.login_required
# def delete(drawing_id):
#     conn = sqlite3.connect('db.db')
#     c = conn.cursor()
#     c.execute("DELETE FROM drawings WHERE id = ?", (drawing_id,))
#     conn.commit()
#     conn.close()
#     return redirect(url_for('admin'))

# @app.route('/delete_all')
# @auth.login_required
# def delete_all():
#     conn = sqlite3.connect('db.db')
#     c = conn.cursor()
#     c.execute("DELETE FROM drawings")
#     conn.commit()
#     conn.close()
#     return redirect(url_for('admin'))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)