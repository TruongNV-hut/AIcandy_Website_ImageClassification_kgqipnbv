"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import io
import json

# Load the class labels
with open('imagenet_classes.txt', 'r') as f:
    IMAGENET_CLASSES = [line.strip() for line in f]


def load_model(model_path):
    model = mobilenet_v2(pretrained=False)  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  
    model.eval() 
    return model

def get_transform():
    # Define the same transforms used during training
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_image(image_bytes):
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    transform = get_transform()
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    # Get the class name
    predicted_class = IMAGENET_CLASSES[predicted_idx.item()]
    
    return predicted_class