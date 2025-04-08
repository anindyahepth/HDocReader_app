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
import torchvision.transforms.functional as TF


import cv2




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
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB, predicted_text TEXT)''')
    conn.commit()
    conn.close()

init_db()
 
 
 
 

def zoom_out_tensor(img_tensor, zoom_factor):
    """
    Zooms out a torch tensor image.

    Args:
        img_tensor (torch.Tensor): Input image tensor (C, H, W).
        zoom_factor (float): Zoom out factor (e.g., 0.5 for 50% zoom out).

    Returns:
        torch.Tensor: Zoomed out image tensor.
    """

    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if zoom_factor <= 0 or zoom_factor > 1:
        raise ValueError("Zoom factor must be between 0 and 1.")

    _, height, width = img_tensor.shape

    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)

    # Resize the image
    zoomed_tensor = TF.resize(img_tensor, (new_height, new_width), antialias=True)

    # Pad the image to the original size
    pad_top = (height - new_height) // 2
    pad_bottom = height - new_height - pad_top
    pad_left = (width - new_width) // 2
    pad_right = width - new_width - pad_left

    zoomed_padded = TF.pad(zoomed_tensor, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='constant', fill=0)

    return zoomed_padded

 
def zoom_in_tensor(img_tensor, zoom_factor, fill_value: int = 0):
    """
    Zooms in on a torch tensor image by cropping and resizing.

    Args:
        img_tensor: Input image tensor (C, H, W).
        zoom_factor: Zoom in factor (e.g., 1.5 for 150% zoom in). Must be > 1.
        fill_value: The value to fill the padded area with during the initial zoom out (default: 0).

    Returns:
        Zoomed-in image tensor (C, H, W).

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the zoom factor is not greater than 1.
    """
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input 'img_tensor' must be a torch.Tensor.")

    if zoom_factor <= 1:
        raise ValueError("Input 'zoom_factor' must be greater than 1.")

    channels, height, width = img_tensor.shape
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)

    # Calculate the crop parameters
    top = (height - new_height) // 2
    left = (width - new_width) // 2

    #check for edge cases where the crop is out of bounds
    if top < 0 or left < 0:
      raise ValueError("Zoom factor is too large, resulting in negative crop coordinates.")
    # Crop the center of the image
    cropped_tensor = TF.crop(img_tensor, top, left, new_height, new_width)

    # Resize the cropped image to the original size
    zoomed_tensor = TF.resize(cropped_tensor, (height, width), antialias=True)
    return zoomed_tensor

 

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
        
        
        
        
def split_handwritten_page(image_path, output_dir="lines", target_size=(512, 64)):
    """
    Splits a handwritten text page into lines using horizontal projection,
    saves each line as an image, and formats them into torch tensors.

    Args:
        image_path (str): Path to the input JPEG image.
        output_dir (str): Directory to save the line images.
        target_size (tuple): Target (width, height) for each line image.
    """

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
        resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to grayscale and normalize
        resized_gray = resized_img.convert("L")
        np_img = np.array(resized_gray) / 255.0

        # Convert to torch tensor
        tensor_img = torch.from_numpy(np_img).float().unsqueeze(0)

        line_images.append(tensor_img)

        # Save line image
        line_filename = os.path.join(output_dir, f"line_{i}.jpg")
        resized_img.save(line_filename)

    return line_images
    
    

        
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
   
    processed_image = transformation(image)
    image_tensor = processed_image.unsqueeze(0)

    return image_tensor



def create_model_vitmae(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls=96,
                                 img_size=img_size,
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model


def make_predictions(image_path, pth_path= './best_CER_6k.pth', dict_path= './dict_alph'):
    
    
    #'/Users/anindyadey/HTR-app/best_CER.pth' 
    #'/Users/anindyadey/HTR-app/dict_alph
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model_vitmae(nb_cls =96, img_size= [64, 512])
    ckpt = torch.load(pth_path, map_location='cpu', weights_only = True)

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['state_dict_ema'].items():
        if re.search(pattern, k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    
    
    model = model.to(device)
    model.eval()

    
    alpha = dict_from_file_to_list(dict_path)
    converter = utils.CTCLabelConverter(alpha)
    
    line_tensors = split_handwritten_page(image_path) #line tensors
    preds_list = []

    for i,line_tensor in enumerate(line_tensors):
         line_tensor = zoom_in_tensor(line_tensor, zoom_factor =2.0)
         image_tensor = line_tensor.unsqueeze(0)
         print(image_tensor.shape)
         image_tensor = image_tensor.to(device)

         with torch.no_grad():
           preds = model(image_tensor)
           preds = preds.float()
           preds_size = torch.IntTensor([preds.size(1)])
           preds = preds.permute(1, 0, 2).log_softmax(2)
           _, preds_index = preds.max(2)
           preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
           preds_str = converter.decode(preds_index.data, preds_size.data)
           predicted_text = preds_str[0] 
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
       
       print(predicted_text)
       
       
        

       conn = sqlite3.connect('db.db')
       c = conn.cursor()
       c.execute("INSERT INTO drawings (data, predicted_text) VALUES (?, ?)", (data, predicted_text))
       conn.commit()
       conn.close()
       return jsonify({'prediction': predicted_text })
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