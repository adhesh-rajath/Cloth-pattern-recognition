from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import numpy as np
import os
import cv2
import torch
from torchvision import models, transforms
import albumentations as albu
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image
from cloths_segmentation.pre_trained_models import create_model
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class_names = ['chequered', 'dotted', 'floral', 'paisley', 'plain', 'striped', 'zigzagged']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ResNet model for classification
model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(r'C:/Users\Adhesh Rajath\Downloads/final_project/final_proj/best_model1.pth', map_location=device))
model_ft.eval()

# Load the segmentation model
model = create_model("Unet_2020-10-30")
model.eval()

def add_black_border(image, border_size):
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def preprocess_image(image_path):
    img = Image.open(image_path)
    transform = albu.Compose([albu.Resize(224, 224), albu.Normalize()])
    img = transform(image=np.array(img))['image']
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def predict_image(model, input_tensor, class_names):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def remove_non_black_part(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        cv2.imwrite(output_path, cropped_image)
    else:
        cv2.imwrite(output_path, image)

def process_and_predict(image_path):
    image = cv2.imread(image_path)
    border_size = 70
    bordered_image = add_black_border(image, border_size)
    padded_image, pads = pad(bordered_image, factor=32, border=cv2.BORDER_CONSTANT)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    segmented_image = cv2.bitwise_and(bordered_image, bordered_image, mask=mask)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented_image.png')
    cv2.imwrite(output_path, segmented_image)
    remove_non_black_part(output_path, output_path)

    cropped_img = cv2.imread(output_path)
    h, w = cropped_img.shape[:2]
    mid_x, mid_y = w // 2, h // 2
    quadrants = {
        'top_left': cropped_img[0:mid_y, 0:mid_x],
        'top_right': cropped_img[0:mid_y, mid_x:w],
        'bottom_left': cropped_img[mid_y:h, 0:mid_x],
        'bottom_right': cropped_img[mid_y:h, mid_x:w]
    }

    prediction_array = []
    for name, quadrant in quadrants.items():
        quadrant_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{name}_quadrant.jpg')
        cv2.imwrite(quadrant_path, quadrant)
        input_tensor = preprocess_image(quadrant_path)
        predicted_label = predict_image(model_ft, input_tensor, class_names)
        prediction_array.append(predicted_label)

    most_common = Counter(prediction_array).most_common(1)[0][0]
    return most_common, prediction_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        most_common, prediction_array = process_and_predict(file_path)
        return render_template('result.html', most_common=most_common, prediction_array=prediction_array)

@app.route('/capture', methods=['POST'])
def capture():
    file = request.files['webcam']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
        file.save(file_path)
        most_common, prediction_array = process_and_predict(file_path)
        return render_template('result.html', most_common=most_common, prediction_array=prediction_array)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

