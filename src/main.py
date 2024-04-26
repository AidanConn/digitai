from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image, ImageOps

app = Flask(__name__)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 11)

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
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32)
    img = img.view(1, 1, 28, 28)

    with torch.no_grad():
        output = model(img)
        predicted = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100

        if predicted == 10:
            # No recognizable digit (since 10 is not a valid digit)
            return jsonify({'prediction': 'No digit detected', 'confidence': confidence})
        else:
            return jsonify({'prediction': predicted, 'confidence': confidence})


if __name__ == '__main__':
    app.run()