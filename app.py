import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

classes=['no', 'yes']

class MyNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model=nn.Sequential(
        nn.Conv2d(1,32,3,padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(32,64,3,padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Flatten(),
        nn.Linear(64*12*12,128),
        nn.ReLU(),
        nn.Linear(128,2)
    )

  def forward(self,x):
    return self.model(x)
  
model=MyNN()
model.load_state_dict(torch.load('brain_tumor_mri_pytorch.pth'))

model.eval()

st.title("Brain Tumor MRI Prediction")
st.write("Upload an MRI image to predict the presence of a brain tumor.")

uploaded_file=st.file_uploader("Choose an image...", type=["jpg","jpeg","png","webp"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image=transform(image)
    image=image.unsqueeze(0)

    with torch.no_grad():
        outputs=model(image)
        _,predicted=torch.max(outputs,1)
        st.write(f'Chance of Brain Tumor: {classes[predicted.item()]}')