import streamlit as st
import torch
import cv2

st.write("App is running!")
st.write(f"PyTorch version: {torch.__version__}")
st.write(f"OpenCV version: {cv2.__version__}")