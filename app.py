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

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV format (BGR)
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
    except Exception as e:
        st.error(f"Error loading labels.json: {e}")
        st.stop()

    # Load database of embeddings
    try:
        database = torch.load("database.pt")
        # Debugging: Display the type and structure of database
        st.write(f"Debug: Type of database after loading: {type(database)}")
        if isinstance(database, tuple):
            st.write(f"Debug: Database is a tuple with {len(database)} elements")
            for i, item in enumerate(database):
                st.write(f"Debug: Element {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
            # Extract the first tensor found in the tuple
            for item in database:
                if isinstance(item, torch.Tensor):
                    database = item
                    st.write("Debug: Extracted tensor from tuple")
                    break
            else:
                st.error("No tensor found in the database tuple.")
                st.stop()
        elif not isinstance(database, torch.Tensor):
            st.error(f"Database is neither a tensor nor a tuple containing a tensor: {type(database)}")
            st.stop()
        # At this point, database should be a tensor
        st.write(f"Debug: Database is now a tensor with shape {database.shape}")
    except Exception as e:
        st.error(f"Error loading database.pt: {e}")
        st.stop()

    # Ensure database is 2D (expected shape: [num_samples, embedding_dim])
    if database.dim() != 2:
        if database.dim() == 1:
            database = database.unsqueeze(0)  # Convert 1D to 2D
            st.write("Debug: Adjusted database from 1D to 2D")
        else:
            st.error(f"Database tensor must be 2D, but has shape {database.shape}")
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
    # Assuming output is a mask with shape [1, H, W]
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

    # Run classification model to get embedding
    with torch.no_grad():
        output = classification_model(image_cls)
    if isinstance(output, tuple):
        embedding = output[0]  # Assume embedding is the first element
        st.write("Debug: Model output was a tuple, extracted first element")
    else:
        embedding = output

    # Ensure embedding is a tensor
    if not isinstance(embedding, torch.Tensor):
        st.error(f"Model output is not a tensor: {type(embedding)}")
        st.stop()

    # Ensure embedding is 2D (expected shape: [1, embedding_dim])
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
        st.write("Debug: Adjusted embedding from 1D to 2D")
    elif embedding.dim() != 2:
        st.error(f"Embedding tensor must be 2D, but has shape {embedding.shape}")
        st.stop()

    # Compute cosine similarity
    try:
        similarities = F.cosine_similarity(embedding, database, dim=1)
    except Exception as e:
        st.error(f"Error computing cosine similarity: {e}")
        st.stop()

    # Get predicted class and confidence
    predicted_class_idx = torch.argmax(similarities).item()
    confidence = similarities[predicted_class_idx].item()
    predicted_class = class_labels[str(predicted_class_idx)]

    # Display classification results
    st.write("### Classification Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")