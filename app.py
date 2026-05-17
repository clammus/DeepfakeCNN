import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build model architecture (same as best config)
def build_model():
    conv_layers = []
    in_channels = 3
    out_channels = 32

    for i in range(6):  # num_conv_layers = 6
        conv_layers += [
            nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.0),
            nn.MaxPool2d(kernel_size=4, stride=2)
        ]
        in_channels = out_channels
        out_channels = min(out_channels * 2, 512)

    # Calculate flat size with dummy input
    dummy = torch.zeros(1, 3, 256, 256)
    dummy_model = nn.Sequential(*conv_layers)
    dummy_out = dummy_model(dummy)
    flat_size = dummy_out.view(1, -1).shape[1]

    model = nn.Sequential(
        *conv_layers,
        nn.Flatten(),
        nn.Linear(flat_size, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 2)
    )
    return model

# Load model
model = build_model()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction function
def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img)
        temperature = 10.0
        outputs = outputs / temperature
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    label = 'Fake' if predicted.item() == 0 else 'Real'
    score = confidence.item()
    
    return {
        'Fake': float(probs[0][0]),
        'Real': float(probs[0][1])
    }

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload Face Image'),
    outputs=gr.Label(num_top_classes=2, label='Prediction'),
    title='Deepfake Detection',
    description='Upload a face image to detect whether it is Real or Fake.'
)

demo.launch()