import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import json

# Title and description
st.title("Fishial App: Fish Segmentation and Classification")
st.write("Upload an image of a fish to get segmentation and classification results.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV format (BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Load segmentation model (Fish Segmentation Model)
    try:
        segmentation_model = torch.jit.load("seg_model.ts")  # Update
        segmentation_model.eval()
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        st.stop()

    # Load classification model (Fish Classification Model)
    try:
        classification_model = torch.jit.load("class_model.ts")  # Update
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        st.stop()

    # Load class labels
    try:
        with open("labels.json", "r") as f:  # Update 
            class_labels = json.load(f)
    except Exception as e:
        st.error(f"Error loading labels.json: {e}")
        st.stop()

    # Preprocess image for segmentation (assuming input size 416x416)
    transform_seg = T.Compose([
        T.ToTensor(),
        T.Resize((416, 416)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_seg = transform_seg(image_cv).unsqueeze(0)  # Add batch dimension

    # Run segmentation model
    with torch.no_grad():
        seg_output = segmentation_model(image_seg)
    # Assuming the output is a mask with shape [1, H, W]
    mask = seg_output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold to binary mask

    # Display segmentation mask
    st.image(mask, caption="Segmentation Mask", use_column_width=True, clamp=True)

    # Preprocess image for classification (assuming input size 224x224)
    transform_cls = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_cls = transform_cls(image_cv).unsqueeze(0)  # Add batch dimension

    # Run classification model
    with torch.no_grad():
        cls_output = classification_model(image_cls)
    probs = torch.softmax(cls_output, dim=1)
    predicted_class_idx = torch.argmax(probs, dim=1).item()
    predicted_class = class_labels[str(predicted_class_idx)]  # Assuming labels.json is a dict
    confidence = probs[0, predicted_class_idx].item()

    # Display classification results
    st.write("### Classification Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")