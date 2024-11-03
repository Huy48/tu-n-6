import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch.nn as nn

# Định nghĩa model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình đã huấn luyện
model = torch.load("MLP_dress.pth")
model.eval()

# Chuyển đổi ảnh trước khi dự đoán
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file.stream).convert("L")  # Chuyển đổi ảnh sang grayscale
        image = transform_image(image)

        # Dự đoán
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        return render_template('result.html', prediction=predicted.item())

if __name__ == "__main__":
    app.run(debug=True)
