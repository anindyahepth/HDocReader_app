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



# def apply_dynamic_quantization(model):
#     """Apply dynamic quantization to the model"""
#     print("Applying dynamic quantization...")

#     # Dynamic quantization - quantizes weights, activations computed in fp32
#     quantized_model = torch.quantization.quantize_dynamic(
#         model,
#         {torch.nn.Linear},  # Quantize Linear layers
#         dtype=torch.qint8
#     )

#     return quantized_model


def recognize_text_with_yolo(image, processor,model,device,
yolo_weights_path: str = '/Users/anindyadey/Dropbox/ML/HDocReader_workspace/HDocReader_app/yolo_weights/best.pt', 
confidence_threshold: float = 0.5,
nms_iou_threshold: float = 0.2) -> dict:

    """
    Performs YOLO object detection at line-level and then transcribes the detected text regions with TrOCR.
    """
    from ultralytics import YOLO

    detector = YOLO(yolo_weights_path)
    print(f"YOLO detector loaded from {yolo_weights_path}")
    
    results = detector(image, iou = nms_iou_threshold, verbose=False) # verbose=False to clean up output
    
    transcriptions = []

    # DETECTION AND SORTING
    
    # Process results from the YOLO model
    for result in results:
        # Get bounding boxes [xmin, ymin, xmax, ymax] and confidence scores
        boxes = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        
        # Combine detections and filter by confidence
        detections = []
        for box, conf in zip(boxes, confs):
            if conf >= confidence_threshold:
                # Add a temporary y-center for vertical sorting
                y_center = (box[1] + box[3]) / 2
                detections.append((y_center, box[0], box[1], box[2], box[3], conf))
        
        # Sort detections primarily by y-center (top to bottom) 
        # and secondarily by xmin (left to right) for correct reading order
        detections.sort(key=lambda x: (x[0], x[1])) 
        
        print(f"Detected {len(detections)} text lines with confidence > {confidence_threshold}. Starting transcription...")

        # Transcribe the detected text regions with TrOCR
        count=0
        for y_center, xmin, ymin, xmax, ymax, conf in detections:
            count+=1
            # Crop the bounding box (ensure coordinates are integers for Pillow)
            cropped_img = image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
            output_dir = './cropped_images_yolo'
            os.makedirs(output_dir, exist_ok=True)
            line_filename = os.path.join(output_dir, f"line_yolo_{count}.jpg")
            cropped_img.save(line_filename)
            
            # Prepare the image for the TrOCR model
            pixel_values = processor(images=cropped_img, return_tensors="pt").pixel_values.to(device)
            
            # Generate the transcription (inference)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            
            # Decode the generated tokens
            transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Store the result
            transcriptions.append({
                "box": [xmin, ymin, xmax, ymax],
                "text": transcribed_text,
                "confidence": conf
            })
            
            print(f"  Confidence: {conf:.2f} - Text: {transcribed_text}")
    
    # RETURN RESULTS AS A DICTIONARY : get ouput['text']
    return {
        "full_text": " ".join([t['text'] for t in transcriptions]),
        "lines": transcriptions
    }

def make_predictions(image_path):

  # Load the TrOCR model and processor - replace this with the fine-tuned model
  processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
  model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

  # Set the model to evaluation mode
  model.eval()

  # Use GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  
  # Load the full image for YOLO detection
  full_image = Image.open(image_path).convert('RGB')
  print(f"Processing full image: {image_path}")
  print(f"Image size: {full_image.size}, mode: {full_image.mode}")
  
  # Use YOLO to detect text regions in the full image
  yolo_result = recognize_text_with_yolo(full_image, processor=processor, model=model, device=device, confidence_threshold=0.3)
  
  # Extract the full text from YOLO results
  if yolo_result and 'full_text' in yolo_result and yolo_result['full_text'].strip():
      print(f"YOLO detected text: {yolo_result['full_text']}")
      # If we have individual line detections, use them
      if yolo_result.get('lines') and len(yolo_result['lines']) > 0:
          preds_list = [line['text'] for line in yolo_result['lines']]
          print(f"Using {len(preds_list)} YOLO-detected text regions")
      else:
          preds_list = [yolo_result['full_text']]
  else:
      # Fallback to line splitting if YOLO fails or detects nothing
      print("YOLO detection failed or no text detected, falling back to line splitting...")
      line_images = split_handwritten_page(image_path)
      preds_list = []
      
      for i, line_image in enumerate(line_images):
          with torch.no_grad():
              predicted_text = recognize_text(line_image, processor=processor)
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

def correct_transcript_with_gemini(gemini_model,draft_transcript: str, image_path: str) -> str:
    if gemini_model is None:
        return draft_transcript
    try:
        image = Image.open(image_path)
        prompt_parts = [
            "You are an expert transcriber specializing in handwritten documents and accurate optical character recognition (OCR).",
            "Review the following draft transcript of a single line of handwritten text.",
            "Using the provided image as the authoritative source, meticulously correct any errors, omissions, or misinterpretations in the draft.",
            "Pay extremely close attention to spelling, punctuation, capitalization, and spacing exactly as it appears in the handwritten image.",
            "Do not correct spelling errors if they are present in the image.",
            "If the draft is a single word, provide the full correct transcription based on the image.",
            "If the draft is entirely incorrect or misses major parts, provide the full correct transcription based on the image.",
            "If the draft is mostly correct, make only the necessary minor corrections.",
            "Do NOT add any explanations or additional text; only provide the corrected transcript.",
            "\n\n**Draft Transcript:**\n",
            f"{draft_transcript}\n\n",
            "**Image Context:**\n",
            image,
            "\n\n**Corrected Transcript:**\n",
        ]
        response = gemini_model.generate_content(prompt_parts)
        corrected_transcript = response.text.strip()
        return corrected_transcript or draft_transcript
    except Exception as e:
        print(f"Error during transcript correction: {e}")
        return draft_transcript


# Function to split a handwritten page into lines - used as a fallback when YOLO fails
def split_handwritten_page(image_path, output_dir="lines", target_size=(512, 64)):

    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")

    # Preprocessing: Binarization
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal Projection
    horizontal_projection = np.sum(binary_img, axis=1)
    max_projection = np.max(horizontal_projection)
    print(f"Max horizontal projection: {max_projection}")

    # Find line boundaries using a more robust approach
    line_starts = []
    line_ends = []
    
    # Smooth the projection to reduce noise
    from scipy import ndimage
    try:
        smoothed_projection = ndimage.gaussian_filter1d(horizontal_projection.astype(float), sigma=3)
    except ImportError:
        # Fallback if scipy not available
        smoothed_projection = horizontal_projection.astype(float)
    
    # Find local minima (valleys) in the projection - these indicate line breaks
    from scipy.signal import find_peaks
    try:
        # Invert the projection to find valleys as peaks
        inverted_projection = max_projection - smoothed_projection
        
        # Find peaks in the inverted projection (valleys in original)
        peaks, properties = find_peaks(inverted_projection, 
                                     height=max_projection * 0.3,  # Minimum valley depth
                                     distance=20,  # Minimum distance between valleys
                                     prominence=max_projection * 0.2)  # Minimum prominence
        
        print(f"Found {len(peaks)} potential line breaks at positions: {peaks}")
        
        # Convert valleys to line boundaries
        if len(peaks) > 0:
            # Start from the beginning
            line_starts.append(0)
            
            # Add line breaks
            for peak in peaks:
                line_ends.append(peak)
                line_starts.append(peak + 1)
            
            # End at the bottom
            line_ends.append(len(smoothed_projection))
            
            # Remove any lines that are too short
            filtered_starts = []
            filtered_ends = []
            for start, end in zip(line_starts, line_ends):
                if end - start > 15:  # Minimum line height
                    filtered_starts.append(start)
                    filtered_ends.append(end)
            
            line_starts = filtered_starts
            line_ends = filtered_ends
            
        else:
            # Fallback: use threshold-based approach
            print("No clear valleys found, using threshold approach...")
            threshold = max_projection * 0.1  # 10% of max projection
            print(f"Using threshold: {threshold}")
            
            text_regions = []
            in_text = False
            start_y = 0
            
            for y, projection_value in enumerate(smoothed_projection):
                if projection_value > threshold and not in_text:
                    start_y = y
                    in_text = True
                elif projection_value <= threshold and in_text:
                    if y - start_y > 20:  # Higher minimum height
                        text_regions.append((start_y, y))
                    in_text = False
            
            if in_text and len(smoothed_projection) - start_y > 20:
                text_regions.append((start_y, len(smoothed_projection)))
            
            for start_y, end_y in text_regions:
                line_starts.append(start_y)
                line_ends.append(end_y)
                
    except ImportError:
        # Fallback if scipy not available
        print("SciPy not available, using simple threshold approach...")
        threshold = max_projection * 0.15  # 15% of max projection
        print(f"Using threshold: {threshold}")
        
        text_regions = []
        in_text = False
        start_y = 0
        
        for y, projection_value in enumerate(smoothed_projection):
            if projection_value > threshold and not in_text:
                start_y = y
                in_text = True
            elif projection_value <= threshold and in_text:
                if y - start_y > 25:  # Higher minimum height
                    text_regions.append((start_y, y))
                in_text = False
        
        if in_text and len(smoothed_projection) - start_y > 25:
            text_regions.append((start_y, len(smoothed_projection)))
        
        for start_y, end_y in text_regions:
            line_starts.append(start_y)
            line_ends.append(end_y)

    print(f"Found {len(line_starts)} line starts: {line_starts}")
    print(f"Found {len(line_ends)} line ends: {line_ends}")

    # Validate that we found lines
    if len(line_starts) == 0:
        print("⚠️ No lines detected! Using entire image as single line.")
        line_starts = [0]
        line_ends = [img.shape[0]]

    line_images = []

    for i, (start_y, end_y) in enumerate(zip(line_starts, line_ends)):
        line_img = img[start_y:end_y, :]
        print(f"Line {i}: y={start_y}-{end_y}, height={end_y-start_y}")

        # Convert to PIL and resize to target size
        pil_img = Image.fromarray(line_img)
        resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        line_images.append(resized_img)

        # Save line image
        line_filename = os.path.join(output_dir, f"line_{i}.jpg")
        resized_img.save(line_filename)
        print(f"Saved line {i} to {line_filename}")

    print(f"Successfully split into {len(line_images)} lines")
    return line_images    


# Function to recognize text if YOLO fails
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
