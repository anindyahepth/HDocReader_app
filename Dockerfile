# Use a Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

#Load local packages

#VOLUME /Users/anindyadey/Library/Caches/pip:/root/.cache/pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenGL libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    python3-opencv \
    pkg-config


# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 8080

# Run the Flask application
CMD ["python", "main.py"]



#Docker commands

#1. docker build -t [tag-1]/[tag-2] .

#2. docker run -p 8080:8080 [tag-1]/[tag-2]