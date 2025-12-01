from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import io

class_names=['Church', 'Enough or Satisfied', 'Friend', 'Love', 'Me', 'Mosque', 'Seat', 'Temple', 'You']

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        self.pool3 = nn.MaxPool2d(2, 2)

        flattened_size = 128 * 25 * 25  # = 80,000

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

net=NeuralNet().to(device)

state_dict=torch.load("model.pth", map_location=device)

net.load_state_dict(state_dict)

net.eval()

transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image_bytes):
    image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image=transform(image).unsqueeze(0)
    return image.to(device)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    try:
        image_bytes=await file.read()
        img=transform_image(image_bytes)
        
        with torch.no_grad():
            outputs=net(img)
            _, predicted=torch.max(outputs, 1)
            
        prediction=class_names[predicted.item()]
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/")
def home():
    return {"message": "Sign Language system running!"}