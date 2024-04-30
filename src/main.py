from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import re
import base64
import io  # Import the io module

app = Flask(__name__)


# Load the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 11)  # 10 digits + 1 for no digit detected

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Model()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()


def preprocess_image(image_data):
    # Remove metadata from image data string
    base64_image = re.sub('^data:image/.+;base64,', '', image_data)
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_image)
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Invert image colors
    image = ImageOps.invert(image)
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Convert to PyTorch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    return image_tensor


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json['image_data']
        # Debugging
        print(image_data[:100])
        image_tensor = preprocess_image(image_data)
        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100
            # Debugging
            print("Predicted:", predicted)
            print("Confidence:", confidence)
            if predicted == 10:
                return jsonify({'prediction': 'No digit detected', 'confidence': confidence})
            else:
                return jsonify({'prediction': predicted, 'confidence': confidence})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
