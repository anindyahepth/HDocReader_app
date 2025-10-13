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

  # Load the TrOCR model and processor - replace this with the fine-tuned model
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
