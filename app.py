import os
from flask import Flask, render_template, request, flash, redirect, url_for, session , redirect
from pymongo import MongoClient
from fileinput import filename
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
])
class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = CnnModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model_state_dict.pth', map_location=device))
model.to(device)
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/Customer_email_pass_data"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Customer_email_pass.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = 'static/uploads'
connection = MongoClient('localhost', 5000)
db = connection['CNNdata']
collection_1 = db['userdata']

app = Flask(__name__)



@app.route('/')
def hello_world():  # put application's code here
    return render_template("homepage1.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=5001 , debug=True)
