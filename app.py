import os

from flask import Flask, render_template, request, flash, redirect, url_for, session , redirect
from pymongo import MongoClient
from fileinput import filename
from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo
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
mongo = PyMongo(app)
connection = MongoClient('localhost', 5000)
db = connection['CNNdata']
collection_1 = db['userdata']

app = Flask(__name__)



@app.route('/')
def hello_world():  # put application's code here
    return render_template("homepage1.html")

@app.route('/clickSignup' , methods=['GET', 'POST'])
def clicksignUp():

       return render_template("signUp.html")

@app.route('/clicklogin' , methods=['GET', 'POST'])
def clicklogin():

        return render_template("Login.html")

@app.route('/profilePicture' , methods=['GET', 'POST'])

def profilePicture():
    file = request.files['profilePic']
    filename = file.filename
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)

    def preprocess(path):
        image = Image.open(path).convert('RGB')
        image = transform(image)
        return image

    def predict(image):
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    image = preprocess(filepath)
    image = image.unsqueeze(0).to(device)
    prediction = predict(image)
    if prediction == 0:
        result = "You have no Disease!"
    else:
        result = "You have Pneumonia, Please consult a doctor"
    os.remove(filepath)
    return render_template('result.html', result=result)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    username = request.form['email']
    password = request.form['pass']
    confirm_password = request.form['cpass']
    data = {
        'User_Email': username,
        'Password': password,
    }

    if password == confirm_password:

            for x in mongo.db.email_pass.find({},{"User_Email":1}):
                if(username == x['User_Email']):
                    return render_template("UserAlreadyExists.html")
            mongo.db.email_pass.insert_one(data)
            return returnform()

    else:
        return "Passwords do not match"

@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form['login-email']
    password = request.form['login-pass']

    for x in mongo.db.email_pass.find({},{"User_Email":1 , "Password":1}):
       if(username == x['User_Email']):
           if(password == x['Password']):
               return "Login Successful"
           else :
               return "Incorrect Password"

       else:
           continue

    return render_template('UsernotFound.html')

@app.route('/returnform' , methods=['GET','POST'])

def returnform():
    return render_template('Info.html')

@app.route('/formfilled' , methods=['GET','POST'])

def formfilled():
    fname = request.form['fname']
    lname = request.form['lname']
    name = fname + " " + lname
    age = request.form['age']
    gender = request.form['rad']

    pec= request.form['o1']
    if(pec == "Yes"):
        pec = pec + "," + request.form['pec']

    all = request.form['o2']
    if(all == "Yes"):
        all = all + "," + request.form['all']

    pxr = request.form['o3']
    if(pxr == "Yes"):
        pxr = pxr + "," + request.form['pxr']

    email = request.form['email1']
    contact = request.form['contact']
    add1 = request.form['add1']
    add2 = request.form['add2']
    address = add1 + add2
    city = request.form['city']
    state = request.form.get('state')
    zip = request.form['zip']

    data = {
        'Name' : name ,
        'Age' : age,
        'Gender' : gender,
        'Pre-Existing Conditions' : pec,
        'Allergy Conditions' : all,
        'Pre X-Ray Details' : pxr,
        'Email' : email,
        'Contact' : contact,
        'Address': address,
        'City' : city,
        'State' : state,
        'Zip' : zip
    }

    mongo.db.UserDetails.insert_one(data)
    f = request.files['profilePic']
    f.save(f.filename)
    return render_template("homepage.html")

@app.route('/hpo' , methods=['GET','POST'])
def hpo():

 return render_template("hpo.html")
@app.route('/Prevention' , methods=['GET','POST'])
def Prevention():
    return render_template("Prevention.html")
@app.route('/Statistics' , methods=['GET','POST'])
def Statistics():
    return render_template("Statistics.html")
@app.route('/ImageUpload', methods=['GET','POST'])
def imageupload():
    return render_template("ImageUpload.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=5001 , debug=True)
