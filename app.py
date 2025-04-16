import streamlit as st
import os
import time
import sys

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Snaily - YOLO Detection",
    page_icon="👁️",
    layout="wide"
)

# Now you can add the debug info and other Streamlit commands
try:
    import cv2
    st.write(f"Python version: {sys.version}")
    st.write(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    st.error(f"OpenCV import error: {str(e)}")
    st.write(f"Python version: {sys.version}")
    
from PIL import Image
import io
import base64
import numpy as np
from utils.detection import process_image, get_model

# Set page config
st.set_page_config(
    page_title="AI Snaily - YOLO Detection",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        color: white;
        background: linear-gradient(90deg, #4b01d7 0%, #8c52ff 100%);
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div {
        background-color: #8c52ff;
    }
    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 5px;
        padding: 30px;
        text-align: center;
    }
    .model-selection {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .results-area {
        margin-top: 30px;
    }
    .image-preview {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🔍 AI Snaily</h1><p>AI Snaily in different YOLO Version</p></div>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## Upload Images")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Drag & drop images here or click to browse",
        type=["jpg", "jpeg", "png", "gif"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Display uploaded images
    if uploaded_files:
        st.markdown("### Selected Files:")
        image_previews = st.container()
        
        with image_previews:
            preview_cols = st.columns(3)
            for i, uploaded_file in enumerate(uploaded_files):
                with preview_cols[i % 3]:
                    st.image(uploaded_file, width=150)
                    st.caption(uploaded_file.name)

with col2:
    st.markdown("## Select YOLO Model")
    
    # Model selection checkboxes
    st.markdown('<div class="model-selection">', unsafe_allow_html=True)
    yolov5 = st.checkbox("YOLOv5", value=True)
    yolov8 = st.checkbox("YOLOv8", value=False)  # Changed default to False to avoid OpenCV errors
    yolov10 = st.checkbox("YOLOv10", value=False)  # Changed default to False
    yolov11 = st.checkbox("YOLOv11", value=False)  # Changed default to False
    st.markdown('</div>', unsafe_allow_html=True)

# Run detection button
if uploaded_files:
    selected_models = []
    if yolov5: selected_models.append("v5")
    if yolov8: selected_models.append("v8")
    if yolov10: selected_models.append("v10")
    if yolov11: selected_models.append("v11")
    
    if not selected_models:
        st.warning("Please select at least one YOLO model.")
    else:
        run_detection = st.button("Run Detection", type="primary")
        
        if run_detection:
            # Create a progress section
            progress_container = st.container()
            
            with progress_container:
                st.markdown("### Processing Images")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Calculate total operations
                total_operations = len(uploaded_files) * len(selected_models)
                completed_operations = 0
                
                # Store results for display
                results = {}
                
                # Process each image with each selected model
                for uploaded_file in uploaded_files:
                    # Save the uploaded file temporarily
                    image_bytes = uploaded_file.getvalue()
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Store results for this image
                    results[uploaded_file.name] = {}
                    
                    for model_version in selected_models:
                        try:
                            # Update status
                            status_text.text(f"Currently processing: {uploaded_file.name} with YOLO{model_version}")
                            
                            # Process the image
                            model = get_model(model_version)
                            result_image = process_image(image, model)
                            
                            # Store the result
                            results[uploaded_file.name][model_version] = result_image
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name} with YOLO{model_version}: {str(e)}")
                            # Store original image as fallback
                            results[uploaded_file.name][model_version] = image
                        
                        # Update progress
                        completed_operations += 1
                        progress_bar.progress(completed_operations / total_operations)
                        time.sleep(0.1)  # Simulate processing time
                
                # Complete
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                time.sleep(1)
                
                # Clear progress section after completion
                progress_container.empty()
                
                # Display results
                st.markdown("## Detection Results")
                
                for image_name, model_results in results.items():
                    st.markdown(f"### {image_name}")
                    
                    # Create columns for each model result
                    result_cols = st.columns(len(model_results))
                    
                    for i, (model_version, result_image) in enumerate(model_results.items()):
                        with result_cols[i]:
                            st.markdown(f"**YOLO{model_version}**")
                            
                            try:
                                # Check if result_image is a numpy array (from OpenCV)
                                if isinstance(result_image, np.ndarray):
                                    # Convert from BGR to RGB if it has 3 channels
                                    if result_image.shape[2] == 3:
                                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                                    pil_image = Image.fromarray(result_image)
                                elif not isinstance(result_image, Image.Image):
                                    # Try to convert to PIL Image if it's not already
                                    pil_image = Image.fromarray(np.array(result_image))
                                else:
                                    pil_image = result_image
                                
                                # Display the image using Streamlit's image function
                                st.image(pil_image, caption=f"YOLO{model_version} Result", use_container_width=True)
                                
                                # Provide a download link for the image
                                buffered = io.BytesIO()
                                pil_image.save(buffered, format="JPEG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                href = f'<a href="data:file/jpg;base64,{img_str}" download="{image_name}_YOLO{model_version}.jpg">Download</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error displaying result for {image_name} with YOLO{model_version}: {str(e)}")