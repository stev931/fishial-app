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
    try:
        segmentation_model = torch.jit.load("seg_model.ts")
        segmentation_model.eval()
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        st.stop()

    # Load classification model
    try:
        classification_model = torch.jit.load("class_model.ts")
        classification_model.eval()
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        st.stop()

    # Load class labels
    try:
        with open("labels.json", "r") as f:
            class_labels = json.load(f)
        # Debug: Inspect class_labels
        st.write(f"Debug: Number of class labels: {len(class_labels)}")
        st.write(f"Debug: First 10 keys in class_labels: {list(class_labels.keys())[:10]}")
    except Exception as e:
        st.error(f"Error loading labels.json: {e}")
        st.stop()

    # Load database tuple
    try:
        database_tuple = torch.load("database.pt")
        # Debug: Inspect database structure
        st.write(f"Debug: Type of database_tuple: {type(database_tuple)}")
        st.write(f"Debug: Number of elements in database_tuple: {len(database_tuple)}")
        for i, item in enumerate(database_tuple):
            st.write(f"Debug: Element {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
            if isinstance(item, list):
                st.write(f"Debug: Element {i} (list) length: {len(item)}, first few entries: {item[:5]}")
            elif isinstance(item, dict):
                st.write(f"Debug: Element {i} (dict) keys: {list(item.keys())[:5]}")

        # Extract embeddings (Element 0)
        database = database_tuple[0]  # Shape [69990, 128]

        # Extract mapping (try Element 1 as a list of class IDs)
        class_ids = database_tuple[1]
        st.write(f"Debug: Length of class_ids: {len(class_ids)}")
        st.write(f"Debug: First 10 class_ids: {class_ids[:10]}")
    except Exception as e:
        st.error(f"Error loading database.pt: {e}")
        st.stop()

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
    st.write(f"Debug: predicted_class_idx: {predicted_class_idx}")

    # Map embedding index to class ID
    try:
        predicted_class_id = class_ids[predicted_class_idx]
        st.write(f"Debug: predicted_class_id: {predicted_class_id}")
        predicted_class = class_labels[str(predicted_class_id)]
    except IndexError:
        st.error(f"Index {predicted_class_idx} out of bounds for class_ids (length {len(class_ids)})")
        st.stop()
    except KeyError as e:
        st.error(f"Class ID {predicted_class_id} not found in class_labels")
        st.stop()

    # Display results
    st.write("### Classification Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {similarities[predicted_class_idx]:.2f}")