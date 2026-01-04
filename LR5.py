import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import requests

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="ResNet-18 Image Classifier",
    page_icon="🔍",
    layout="centered"
)

st.title(" Image Classification with Pre-trained ResNet-18")
st.write("""
This app uses a **pre-trained ResNet-18 model** to classify uploaded images into one of 1,000 classes.  
It runs entirely on CPU and shows the top-5 predictions with confidence scores and a bar chart.
""")

# -----------------------------
# Load model - CPU 
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Preprocessing transforms
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load ImageNet labels
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = [line.strip() for line in response.text.split("\n")]
    return labels

labels = load_imagenet_labels()

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image to classify",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and inference
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    # Softmax + Top-5
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    top5_prob = top5_prob.cpu().numpy()
    top5_idx = top5_idx.cpu().numpy()
    top5_labels = [labels[idx] for idx in top5_idx]

    # Display top-5 predictions in decimal
    st.subheader("🔍 Top-5 Predicted Classes")
    for i in range(5):
        st.write(f"{i+1}. **{top5_labels[i]}** — {top5_prob[i]:.4f}")

    # Bar chart
    st.subheader("📊 Prediction Confidence Bar Chart")

    chart_data = pd.DataFrame({
        "Class": top5_labels,
        "Probability": top5_prob
    })

    st.bar_chart(chart_data.set_index("Class"))

    
else:
    st.info("👆 Please upload an image (JPG,JPEG or PNG) to get started.")

