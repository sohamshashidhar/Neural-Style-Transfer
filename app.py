import streamlit as st
import os
from PIL import Image

# Title
st.title("🎨 AI-Powered Style Transfer")

# Upload images
content_image = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

if content_image and style_image:
    # Display uploaded images
    st.image(content_image, caption="Content Image", use_column_width=True)
    st.image(style_image, caption="Style Image", use_column_width=True)

    # Save images temporarily
    content_path = "content.jpg"
    style_path = "style.jpg"

    with open(content_path, "wb") as f:
        f.write(content_image.getbuffer())

    with open(style_path, "wb") as f:
        f.write(style_image.getbuffer())

    st.success("✅ Images uploaded successfully! Now click 'Run Style Transfer'.")

    # Run the model (we'll implement this next)
    if st.button("Run Style Transfer"):
        os.system("python style_transfer.py")  # Calls the next script
        st.image("output.jpg", caption="Stylized Image", use_column_width=True)
