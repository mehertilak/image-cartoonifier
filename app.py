from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
from PIL import Image
import io
import base64
from cartoonifier import (
    apply_classic_cartoon,
    apply_comic_style,
    apply_watercolor_style,
    apply_3d_animation_style
)

app = Flask(__name__)

# Configure maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        style = request.form.get('style', 'classic')
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        # Apply selected style
        style_functions = {
            'classic': apply_classic_cartoon,
            'comic': apply_comic_style,
            'watercolor': apply_watercolor_style,
            '3d': apply_3d_animation_style
        }
        
        if style not in style_functions:
            return jsonify({'success': False, 'error': 'Invalid style selected'}), 400
        
        # Process image with selected style
        result = style_functions[style](img)
        
        # Convert result to base64 for response
        _, buffer = cv2.imencode('.png', result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")  # Log the error
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download_image():
    try:
        # Get base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Decode base64 image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # Create file-like object
        img_io = io.BytesIO(img_bytes)
        img_io.seek(0)
        
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='cartoonified.png'
        )
        
    except Exception as e:
        print(f"Error downloading image: {str(e)}")  # Log the error
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
