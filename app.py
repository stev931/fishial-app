import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import json
import torch.nn.functional as F

# Title and description
st.title("Fishial App: Fish Segmentation and Classification")
st.write("Upload an image of a fish to get segmentation and classification results.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Load segmentation model
    segmentation_model = torch.jit.load("seg_model.ts")
    segmentation_model.eval()

    # Load classification model
    classification_model = torch.jit.load("class_model.ts")
    classification_model.eval()

    # Load class labels
    with open("labels.json", "r") as f:
        class_labels = json.load(f)

    # Load database tuple
    database_tuple = torch.load("database.pt")
    database = database_tuple[0]  # Embeddings tensor [69990, 128]
    class_ids = database_tuple[1]  # List of class IDs (length 69990)

    # Preprocess for segmentation
    transform_seg = T.Compose([
        T.ToTensor(),
        T.Resize((416, 416)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_seg = transform_seg(image_cv).unsqueeze(0)

    # Run segmentation
    with torch.no_grad():
        seg_output = segmentation_model(image_seg)
    mask = (seg_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    st.image(mask, caption="Segmentation Mask", use_column_width=True, clamp=True)

    # Preprocess for classification
    transform_cls = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_cls = transform_cls(image_cv).unsqueeze(0)

    # Get embedding
    with torch.no_grad():
        output = classification_model(image_cls)
    embedding = output[0] if isinstance(output, tuple) else output
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    # Compute similarity and predict
    similarities = F.cosine_similarity(embedding, database, dim=1)
    predicted_class_idx = torch.argmax(similarities).item()
    predicted_class_id = class_ids[predicted_class_idx]
    predicted_class = class_labels[str(predicted_class_id)]

    # Display results
    st.write("### Classification Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {similarities[predicted_class_idx]:.2f}")