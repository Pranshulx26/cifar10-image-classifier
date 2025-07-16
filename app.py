
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

from model import CIFAR10ModelV4  
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = CIFAR10ModelV4(num_classes=len(class_names))
model.load_state_dict(torch.load("models/cifar10_model_9.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    if request.method == "POST":
        img = request.files["image"]
        if img:
            filename = os.path.join("static", img.filename)
            img.save(filename)

            image = Image.open(filename).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]
    
    return render_template("index.html", prediction=prediction, image_path=filename, classes=class_names)

if __name__ == "__main__":
    app.run(debug=True)
