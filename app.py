import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import json

st.title("Fishial App: Fish Segmentation and Classification")

uploaded_file = st.file_uploader("Upload an image of a fish", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load models and labels (assume files are in the same directory)
    segmentation_model = torch.jit.load("fish_segmentation_resnet18.torchscript")
    segmentation_model.eval()
    classification_model = torch.jit.load("fish_classification_convnext.torchscript")
    classification_model.eval()
    with open("labels.json", "r") as f:
        class_labels = json.load(f)

    # Preprocess and run segmentation
    image_seg = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_seg = image_seg.unsqueeze(0)
    image_seg = torch.nn.functional.interpolate(image_seg, size=(416, 416), mode='bilinear', align_corners=False)
    with torch.no_grad():
        seg_output = segmentation_model(image_seg)
    mask = seg_output[0].argmax(dim=0).cpu().numpy()

    # Preprocess and run classification
    image_cls = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_cls = image_cls.unsqueeze(0)
    image_cls = torch.nn.functional.interpolate(image_cls, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        cls_output = classification_model(image_cls)
    probs = torch.softmax(cls_output, dim=1)
    predicted_class_idx = torch.argmax(probs, dim=1).item()
    predicted_class = class_labels[predicted_class_idx]
    confidence = float(probs[0, predicted_class_idx])

    st.write("### Results")
    st.write(f"Segmentation Mask Shape: {mask.shape}")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")