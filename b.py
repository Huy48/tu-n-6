import torch
from flask import Flask, render_template, request

# Hàm tính toán CrossEntropy Loss
def crossEntropyLoss(output, target):
    output_softmax = softmax(output)
    loss = -torch.sum(target * torch.log(output_softmax))
    return loss

# Hàm tính toán Mean Square Error
def meanSquareError(output, target):
    return torch.mean((output - target) ** 2)

# Hàm tính toán Binary Entropy Loss
def binaryEntropyLoss(output, target, n):
    loss = -(target * torch.log(output) + (1 - target) * torch.log(1 - output))
    return torch.sum(loss) / n

# Các hàm kích hoạt
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

def softmax(zi):
    exp_zi = torch.exp(zi)
    return exp_zi / torch.sum(exp_zi)

def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    if request.method == 'POST':
        inputs = torch.tensor([float(x) for x in request.form['inputs'].split(',')])
        target = torch.tensor([float(y) for y in request.form['target'].split(',')])
        n = len(inputs)

        # Tính toán các loại loss
        mse = meanSquareError(inputs, target)
        binary_loss = binaryEntropyLoss(sigmoid(inputs), target, n)
        cross_loss = crossEntropyLoss(inputs, target)

        # Tính toán các hàm kích hoạt
        f_sigmoid = sigmoid(inputs)
        f_relu = relu(inputs)
        f_softmax = softmax(inputs)
        f_tanh = tanh(inputs)

        return render_template('result1.html', mse=mse.item(), binary_loss=binary_loss.item(), cross_loss=cross_loss.item(),
                               sigmoid=f_sigmoid.numpy(), relu=f_relu.numpy(), softmax=f_softmax.numpy(), tanh=f_tanh.numpy())

if __name__ == "__main__":
    app.run(debug=True)
