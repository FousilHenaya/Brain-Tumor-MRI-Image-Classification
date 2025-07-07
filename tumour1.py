

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd

# Define class labels
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load('C:/Users/Appu/Desktop/data science/python/tumour/final_resnet18_brain_tumor.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# ---- Sidebar Navigation ----
st.sidebar.title("📺 HOME")
page = st.sidebar.radio("Go to", ["🧠 Brain Tumor Classification", "📊 Model Comparison: Custom CNN vs Fine-Tuned ResNet18"])

# Streamlit UI

if page == "🧠 Brain Tumor Classification":
    st.title("🧠 Brain Tumor Classification")
    st.write("Upload an MRI image and the model will predict the type of brain tumor.")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

        st.markdown(f"### 🧠 Tumor Type: `{CLASS_NAMES[pred.item()]}`")
        st.markdown(f"### 🔍 Confidence: `{conf.item()*100:.2f}%`")






if page == "📊 Model Comparison: Custom CNN vs Fine-Tuned ResNet18":
    st.title("📊 Model Comparison: Custom CNN vs Fine-Tuned ResNet18")

# Define evaluation metrics
    comparison_data = {
        "Metric": [
            "Train Accuracy",
            "Validation Accuracy",
            "Test Accuracy",
            "Best F1-Score",
            "Overfitting",
            "Training Time",
           "Model Size",
           "Deployment Suitability"
           ],
        "Custom CNN": [
            "91.6%",
            "23.5%",
            "21.9%",
            "≈ 0.25",
            "Severe overfitting",
            "Fast",
            "Small",
            "Not suitable"
            ],
        "Fine-Tuned ResNet18": [
            "99.7%",
            "95.8%",
            "95.5%",
            "≈ 0.96",
            "Generalizes well",
            "Moderate",
            "Larger",
            "Recommended"
            ]
        }

    # Convert to DataFrame
    df_compare = pd.DataFrame(comparison_data)

    # Display in Streamlit
    st.dataframe(df_compare, use_container_width=True)

    # Optional: Add final recommendation
    st.success("Final Verdict: Fine-Tuned ResNet18 is the most accurate, generalizable, and deployment")