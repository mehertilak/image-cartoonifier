# Image Cartoonifier

A Python application that converts regular images into cartoon-style images using computer vision techniques.

## Features

- Upload any image (supports JPG, PNG, BMP, GIF, TIFF formats)
- Convert images to cartoon style using advanced image processing
- Save the cartoonified images
- User-friendly GUI interface

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python cartoonifier.py
   ```
2. Click "Open Image" to select an image
3. Click "Cartoonify" to convert the image
4. Click "Save Image" to save the cartoonified version

## How it works

The application uses several image processing techniques:
- Bilateral filtering for smoothing while preserving edges
- Edge detection using adaptive thresholding
- Color quantization using K-means clustering
- Combination of edge mask with color-quantized image

## Requirements

- opencv-python
- numpy
- Pillow
