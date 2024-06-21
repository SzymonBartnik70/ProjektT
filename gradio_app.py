import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# Definicja klasy modelu
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Załadowanie modelu
model = SimpleCNN(num_classes=120)  
model.load_state_dict(torch.load('saved_model_epoch_15.pth', map_location=torch.device('cpu')))
model.eval()

# Funkcja do przewidywania
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    return f'Class: {predicted_class}'

# Tworzenie interfejsu Gradio
iface = gr.Interface(fn=predict, inputs="image", outputs="text", title="Klasyfikator Ras Psów")
iface.launch(server_name="0.0.0.0", server_port=7865)


