import os
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
import segmentation_models_pytorch as smp
from PIL import Image

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODEL LOADING ---
# We must define the model EXACTLY as it was trained
# Assuming the auto-configuration from the training script found 1 class (Binary)
# If your model was trained with more classes, change classes=1 to that number
try:
    print("Loading model...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, # We load our own weights
        in_channels=3,
        classes=1, # Primary assumption: Binary segmentation
        activation=None
    )
    
    # Load the trained weights
    model_path = "best_leaf_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("If this is a dimension error, try changing 'classes=1' to the correct number in app.py")

def preprocess_image(image_path):
    """Load and preprocess image for the model"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to training size (256x256)
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize (0-1)
    img_norm = img_resized / 255.0
    
    # Transpose to (Channels, Height, Width) -> (3, 256, 256)
    img_t = img_norm.transpose(2, 0, 1).astype('float32')
    
    # Convert to Tensor and add batch dimension -> (1, 3, 256, 256)
    tensor = torch.from_numpy(img_t).unsqueeze(0).to(DEVICE)
    
    return tensor, img_resized

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Run Inference
        try:
            input_tensor, original_img = preprocess_image(filepath)
            
            with torch.no_grad():
                output = model(input_tensor)
                
                # Apply Sigmoid for binary mask (0-1 probability)
                output = torch.sigmoid(output)
                
                # Threshold at 0.5 to get binary mask (0 or 1)
                mask = (output > 0.5).float()
                
                # Remove batch and channel dims -> (256, 256)
                mask = mask.squeeze().cpu().numpy()
            
            # --- CREATE OVERLAY ---
            # Create a red overlay
            # Mask is 0 or 1. We want red where mask is 1.
            
            # Create a colored mask (Red)
            # OpenCV uses BGR
            colored_mask = np.zeros_like(original_img)
            colored_mask[:, :, 2] = 255 # Red channel set to max
            
            # Convert mask to boolean for indexing
            mask_bool = mask.astype(bool)
            
            # Blend images
            overlay = original_img.copy()
            # Apply alpha blending only where mask is True
            # alpha = 0.5 (50% transparency)
            overlay[mask_bool] = cv2.addWeighted(
                original_img[mask_bool], 0.5, 
                colored_mask[mask_bool], 0.5, 
                0
            ).squeeze() 
            
            # Save result
            result_filename = f"pred_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            
            # Convert back to BGR for saving with OpenCV
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, overlay_bgr)
            
            return jsonify({
                'original': filepath,
                'result': result_path,
                'success': True
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
