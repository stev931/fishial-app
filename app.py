import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import json
import torch.nn.functional as F

st.title("Fishial App: Fish Segmentation and Classification")
st.write("Upload an image of a fish to get segmentation and classification results.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Load models
    segmentation_model = torch.jit.load("seg_model.ts").eval()
    classification_model = torch.jit.load("class_model.ts").eval()

    # Load labels
    with open("labels.json", "r") as f:
        class_labels = json.load(f)

    # Load and process database
    database = torch.load("database.pt")  # [num_classes, max_val, embedding_size]
    num_classes, max_val, embedding_size = database.shape
    database_flat = database.view(-1, embedding_size)
    class_ids = [i for i in range(num_classes) for _ in range(max_val)]

    # Segmentation
    transform_seg = T.Compose([
        T.ToTensor(),
        T.Resize((416, 416)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_seg = transform_seg(image_cv).unsqueeze(0)
    with torch.no_grad():
        seg_output = segmentation_model(image_seg)
    mask = (seg_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    st.image(mask, caption="Segmentation Mask", use_column_width=True, clamp=True)

    # Classification
    transform_cls = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_cls = transform_cls(image_cv).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(image_cls)
    embedding = output[0] if isinstance(output, tuple) else output
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    # Similarity and prediction
    similarities = F.cosine_similarity(embedding, database_flat, dim=1)
    predicted_class_idx = torch.argmax(similarities).item()
    predicted_class_id = class_ids[predicted_class_idx]
    predicted_class = class_labels[str(predicted_class_id)]

    st.write("### Classification Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {similarities[predicted_class_idx]:.2f}")