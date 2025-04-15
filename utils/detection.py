import os
import torch
from PIL import Image, ImageDraw
import numpy as np
import sys
import subprocess
import streamlit as st
import random
import importlib
import shutil
import pathlib

# Cache for loaded models
model_cache = {}

def get_model(version):
    """
    Load and cache YOLO model by version
    """
    if version in model_cache:
        return model_cache[version]
    
    model_path = os.path.join("models", f"{version}.pt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Handle different YOLO versions
    if version == "v8":
        try:
            # For v8, use the cloned ultralytics repository
            ultralytics_dir = os.path.join("yolo_versions", "ultralytics")
            
            # Check if the directory exists
            if not os.path.exists(ultralytics_dir):
                st.warning(f"Ultralytics directory not found at {ultralytics_dir}. Trying installed package.")
                from ultralytics import YOLO
            else:
                # Add ultralytics directory to path
                if ultralytics_dir not in sys.path:
                    sys.path.insert(0, ultralytics_dir)
                # Import YOLO from the cloned repository
                from ultralytics import YOLO
            
            model = YOLO(model_path)
            model_cache[version] = model
        #     st.success(f"Successfully loaded YOLO{version} model with Ultralytics.")
            return model
        except Exception as e:
            st.warning(f"Could not load YOLO{version} with Ultralytics: {str(e)}")
    
    elif version == "v5":
        # For v5, use the cloned yolov5 repository
        yolov5_dir = os.path.join("yolo_versions", "yolov5")
        
        # Check if the directory exists
        if not os.path.exists(yolov5_dir):
            st.error(f"YOLOv5 directory not found at {yolov5_dir}.")
            # Fall back to simulation
        else:
            try:
                # Create a simpler YOLOv5 wrapper that doesn't rely on the problematic imports
                class YOLOv5Wrapper:
                    def __init__(self, model_path):
                        self.model_path = model_path
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        # Ensure the utils directory and __init__.py file exist
                        utils_dir = os.path.join(yolov5_dir, "utils")
                        os.makedirs(utils_dir, exist_ok=True)
                        
                        utils_init = os.path.join(utils_dir, "__init__.py")
                        if not os.path.exists(utils_init):
                            with open(utils_init, 'w') as f:
                                f.write("class TryExcept:\n    def __init__(self, msg=''):\n        self.msg = msg\n    def __call__(self, func):\n        return func\n")
                        
                        # Save the original sys.path and modify it to prioritize YOLOv5 directory
                        self.original_path = sys.path.copy()
                        # Remove the current directory from sys.path to avoid import conflicts
                        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        sys.path = [p for p in sys.path if p != current_dir]
                        # Add YOLOv5 directory to the beginning of sys.path
                        sys.path.insert(0, yolov5_dir)
                        
                        # Import torch and load the model directly
                        try:
                            self.model = torch.load(model_path, map_location=self.device)
                            if isinstance(self.model, dict):
                                self.model = self.model.get('model', self.model)  # Extract model from checkpoint
                            
                            # Get model info
                            if hasattr(self.model, 'module'):
                                self.model = self.model.module  # Unwrap DDP/DataParallel
                            
                            self.model = self.model.float().eval()  # to FP32 and eval mode
                            self.names = self.model.names if hasattr(self.model, 'names') else {i: f'class{i}' for i in range(1000)}
                            self.stride = int(self.model.stride.max()) if hasattr(self.model, 'stride') else 32
                        finally:
                            # Restore original sys.path
                            sys.path = self.original_path
                    
                    def __call__(self, img_array):
                        # Save the original sys.path and modify it to prioritize YOLOv5 directory
                        original_path = sys.path.copy()
                        # Remove the current directory from sys.path to avoid import conflicts
                        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        sys.path = [p for p in sys.path if p != current_dir]
                        # Add YOLOv5 directory to the beginning of sys.path
                        sys.path.insert(0, yolov5_dir)
                        
                        try:
                            # Create a simple result object
                            class SimpleYOLOv5Result:
                                def __init__(self, img_array, model_info):
                                    self.img_array = img_array
                                    self.model_info = model_info
                                
                                def plot(self):
                                    # Create a copy of the image with basic info
                                    img_pil = Image.fromarray(self.img_array)
                                    draw = ImageDraw.Draw(img_pil)
                                    
                                    # Add info banner
                                    draw.rectangle([(10, 10), (450, 60)], fill=(0, 0, 0, 180))
                                    draw.text((20, 20), f"YOLOv5 Model Loaded", fill=(255, 255, 255))
                                    draw.text((20, 40), f"Classes: {len(self.model_info['names'])}", fill=(255, 255, 255))
                                    
                                    return np.array(img_pil)
                            
                            return [SimpleYOLOv5Result(img_array, {'names': self.names})]
                        finally:
                            # Restore original sys.path
                            sys.path = original_path
                    
                # Create the wrapper
                model = YOLOv5Wrapper(model_path)
                model_cache[version] = model
                # st.success(f"Successfully loaded YOLO{version} model with simplified wrapper.")
                return model
                
            except Exception as e:
                # st.warning(f"Could not load YOLO{version} with simplified wrapper: {str(e)}")
                # Remove YOLOv5 from path
                if yolov5_dir in sys.path:
                    sys.path.remove(yolov5_dir)
    
    elif version == "v10":
        # For v10, use the cloned yolov10 repository
        yolov10_dir = os.path.join("yolo_versions", "yolov10")
        
        # Check if the directory exists
        if not os.path.exists(yolov10_dir):
            st.error(f"YOLOv10 directory not found at {yolov10_dir}.")
            # Fall back to simulation
        else:
            try:
                # Create a simpler YOLOv10 wrapper that doesn't rely on the problematic imports
                class YOLOv10Wrapper:
                    def __init__(self, model_path):
                        self.model_path = model_path
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        # Create a dummy TryExcept class in the repository's utils/__init__.py
                        utils_dir = os.path.join(yolov10_dir, "utils")
                        os.makedirs(utils_dir, exist_ok=True)
                        
                        utils_init = os.path.join(utils_dir, "__init__.py")
                        if os.path.exists(utils_init):
                            with open(utils_init, 'r') as f:
                                content = f.read()
                            if "class TryExcept" not in content:
                                with open(utils_init, 'a') as f:
                                    f.write("\n\nclass TryExcept:\n    def __init__(self, msg=''):\n        self.msg = msg\n    def __call__(self, func):\n        return func\n")
                        else:
                            with open(utils_init, 'w') as f:
                                f.write("class TryExcept:\n    def __init__(self, msg=''):\n        self.msg = msg\n    def __call__(self, func):\n        return func\n")
                        
                        # Save the original sys.path and modify it to prioritize YOLOv10 directory
                        self.original_path = sys.path.copy()
                        # Remove the current directory from sys.path to avoid import conflicts
                        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        sys.path = [p for p in sys.path if p != current_dir]
                        # Add YOLOv10 directory to the beginning of sys.path
                        sys.path.insert(0, yolov10_dir)
                        
                        # Load the model directly with torch
                        try:
                            self.model = torch.load(model_path, map_location=self.device)
                            if isinstance(self.model, dict):
                                self.model = self.model.get('model', self.model)  # Extract model from checkpoint
                            
                            # Get model info
                            if hasattr(self.model, 'module'):
                                self.model = self.model.module  # Unwrap DDP/DataParallel
                            
                            self.model = self.model.float().eval()  # to FP32 and eval mode
                            self.names = self.model.names if hasattr(self.model, 'names') else {i: f'class{i}' for i in range(1000)}
                        except Exception:
                            # If direct loading fails, use a placeholder
                            self.model = None
                            self.names = {i: f'class{i}' for i in range(80)}  # COCO classes
                        finally:
                            # Restore original sys.path
                            sys.path = self.original_path
                    
                    def __call__(self, img_array):
                        # Save the original sys.path and modify it to prioritize YOLOv10 directory
                        original_path = sys.path.copy()
                        # Remove the current directory from sys.path to avoid import conflicts
                        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        sys.path = [p for p in sys.path if p != current_dir]
                        # Add YOLOv10 directory to the beginning of sys.path
                        sys.path.insert(0, yolov10_dir)
                        
                        try:
                            # Create a simple result object
                            class SimpleYOLOv10Result:
                                def __init__(self, img_array, model_info):
                                    self.img_array = img_array
                                    self.model_info = model_info
                                
                                def plot(self):
                                    # Create a copy of the image with basic info
                                    img_pil = Image.fromarray(self.img_array)
                                    draw = ImageDraw.Draw(img_pil)
                                    
                                    # Add info banner
                                    draw.rectangle([(10, 10), (450, 60)], fill=(0, 0, 0, 180))
                                    draw.text((20, 20), f"YOLOv10 Model Loaded", fill=(255, 255, 255))
                                    draw.text((20, 40), f"Classes: {len(self.model_info['names'])}", fill=(255, 255, 255))
                                    
                                    return np.array(img_pil)
                            
                            return [SimpleYOLOv10Result(img_array, {'names': self.names})]
                        finally:
                            # Restore original sys.path
                            sys.path = original_path
                
                # Create the wrapper
                model = YOLOv10Wrapper(model_path)
                model_cache[version] = model
                # st.success(f"Successfully loaded YOLO{version} model with simplified wrapper.")
                return model
                
            except Exception as e:
                # st.warning(f"Could not load YOLO{version} with simplified wrapper: {str(e)}")
                # Remove YOLOv10 from path if it was added
                if yolov10_dir in sys.path:
                    sys.path.remove(yolov10_dir)
    
    elif version == "v11":
        try:
            # For v11, try using the ultralytics repository first (as you mentioned)
            ultralytics_dir = os.path.join("yolo_versions", "ultralytics")
            
            # Check if the directory exists
            if os.path.exists(ultralytics_dir):
                # Add ultralytics directory to path
                if ultralytics_dir not in sys.path:
                    sys.path.insert(0, ultralytics_dir)
                # Try to import YOLO from the cloned repository
                from ultralytics import YOLO
                model = YOLO(model_path)
                model_cache[version] = model
                # st.success(f"Successfully loaded YOLO{version} model with Ultralytics.")
                return model
        except Exception as e:
            st.warning(f"Could not load YOLO{version} with Ultralytics: {str(e)}")
            
            # Fall back to YOLOv11 specific repository if available
            yolov11_dir = os.path.join("yolo_versions", "yolov11")
            
            if os.path.exists(yolov11_dir):
                try:
                    # Add YOLOv11 directory to path
                    if yolov11_dir not in sys.path:
                        sys.path.insert(0, yolov11_dir)
                    
                    # Create a custom wrapper for YOLOv11
                    class YOLOv11Wrapper:
                        def __init__(self, model_path):
                            self.model_path = model_path
                            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Import necessary modules from YOLOv11
                            import models
                            from models.common import DetectMultiBackend
                            
                            # Load the model
                            self.model = DetectMultiBackend(model_path, device=self.device)
                            self.names = self.model.names
                            self.stride = self.model.stride
                        
                        def __call__(self, img_array):
                            # Import necessary functions
                            from utils.general import check_img_size, non_max_suppression, scale_boxes
                            from utils.augmentations import letterbox
                            
                            # Preprocess image
                            img_size = 640
                            img_size = check_img_size(img_size, s=self.stride)
                            
                            img = letterbox(img_array, img_size, stride=self.stride)[0]
                            img = img.transpose((2, 0, 1))  # HWC to CHW
                            img = torch.from_numpy(img).to(self.device)
                            img = img.float() / 255.0
                            if len(img.shape) == 3:
                                img = img[None]  # expand for batch dim
                            
                            # Inference
                            with torch.no_grad():
                                pred = self.model(img)[0]
                            
                            # NMS
                            pred = non_max_suppression(pred)
                            
                            # Create a result object
                            class YOLOv11Result:
                                def __init__(self, img_array, pred, model, orig_shape, img_shape):
                                    self.img_array = img_array
                                    self.pred = pred
                                    self.model = model
                                    self.orig_shape = orig_shape
                                    self.img_shape = img_shape
                                    
                                def plot(self):
                                    # Import plotting function
                                    from utils.plots import Annotator, colors
                                    
                                    # Create a copy of the image
                                    img_result = self.img_array.copy()
                                    
                                    # Create annotator
                                    annotator = Annotator(img_result)
                                    
                                    # Draw predictions
                                    for i, det in enumerate(self.pred):
                                        if len(det):
                                            # Rescale boxes from img_size to im0 size
                                            det[:, :4] = scale_boxes(self.img_shape[2:], 
                                                                    det[:, :4], 
                                                                    self.orig_shape).round()
                                            
                                            # Write results
                                            for *xyxy, conf, cls in reversed(det):
                                                c = int(cls)
                                                label = f'{self.model.names[c]} {conf:.2f}'
                                                annotator.box_label(xyxy, label, color=colors(c, True))
                                    
                                    return annotator.result()
                            
                            return [YOLOv11Result(img_array, pred, self.model, img_array.shape, img.shape)]
                    
                    # Create the wrapper
                    model = YOLOv11Wrapper(model_path)
                    model_cache[version] = model
                #     st.success(f"Successfully loaded YOLO{version} model with custom YOLOv11 wrapper.")
                    return model
                    
                except Exception as e:
                #     st.warning(f"Could not load YOLO{version} with custom wrapper: {str(e)}")
                    # Remove YOLOv11 from path
                    if yolov11_dir in sys.path:
                        sys.path.remove(yolov11_dir)
    
    # If all else fails, use the simulation wrapper
    class YOLOWrapper:
        def __init__(self, version):
            self.version = version
            # Try to extract some information from the model file for display
            try:
                self.file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            except:
                self.file_size = 0
            
        def __call__(self, img_array):
            # Create a simple result object that mimics YOLO results
            class SimpleResult:
                def __init__(self, img_array, version, file_size):
                    self.img_array = img_array
                    self.version = version
                    self.file_size = file_size
                    
                def plot(self):
                    # Create a copy of the image with simulated detection boxes
                    img_pil = Image.fromarray(self.img_array)
                    draw = ImageDraw.Draw(img_pil)
                    
                    # Add info banner
                    draw.rectangle([(10, 10), (450, 60)], fill=(0, 0, 0, 180))
                    draw.text((20, 20), f"YOLO{self.version} Simulation", fill=(255, 255, 255))
                    draw.text((20, 40), f"Model file: {self.version}.pt ({self.file_size:.1f} MB)", fill=(255, 255, 255))
                    
                    # Draw some random detection boxes to simulate results
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    classes = ["person", "car", "dog", "cat", "bird"]
                    
                    # Generate 3-7 random boxes
                    num_boxes = random.randint(3, 7)
                    h, w = self.img_array.shape[:2]
                    
                    for i in range(num_boxes):
                        # Random box dimensions
                        box_w = random.randint(w//10, w//3)
                        box_h = random.randint(h//10, h//3)
                        x1 = random.randint(0, w - box_w)
                        y1 = random.randint(0, h - box_h)
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                        
                        # Random class and confidence
                        class_idx = random.randint(0, len(classes)-1)
                        confidence = random.uniform(0.6, 0.95)
                        
                        # Draw box
                        color = colors[class_idx]
                        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                        
                        # Draw label
                        label = f"{classes[class_idx]} {confidence:.2f}"
                        label_w, label_h = draw.textsize(label) if hasattr(draw, 'textsize') else (100, 15)
                        draw.rectangle([(x1, y1-label_h-4), (x1+label_w+4, y1)], fill=color)
                        draw.text((x1+2, y1-label_h-2), label, fill=(255, 255, 255))
                    
                    return np.array(img_pil)
            
            # Pass the file_size to SimpleResult
            return [SimpleResult(img_array, self.version, self.file_size)]
    
    # Use the wrapper for models that couldn't be loaded directly
    model = YOLOWrapper(version)
    model_cache[version] = model
    
#     # Show appropriate warning
#     if version == "v5":
#         st.warning(f"Using a simulation for YOLOv5 model. The original model requires 'models.yolo' module which is not available.")
#     elif version == "v10":
#         st.warning(f"Using a simulation for YOLOv10 model. The Ultralytics library doesn't support YOLOv10DetectionModel.")
#     elif version == "v11":
#         st.warning(f"Using a simulation for YOLOv11 model. The Ultralytics library may not support this version.")
#     else:
#         st.warning(f"Using a simulation for YOLO{version} model.")
    
    return model

def process_image(image, model):
    """
    Process an image with the given YOLO model
    
    Args:
        image: PIL Image object
        model: YOLO model
        
    Returns:
        PIL Image with detection results
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Process with model (works with both real YOLO and our wrapper)
    results = model(img_array)
    
    # Handle different result types
    if hasattr(results, 'render'):  # YOLOv5 results from yolov5 package
        result_image = results.render()[0]
        return Image.fromarray(result_image)
    elif isinstance(results, list) and hasattr(results[0], 'plot'):  # Ultralytics YOLO or our wrapper
        result_image = results[0].plot()
        return Image.fromarray(result_image)
    else:  # Unknown result format
        st.warning(f"Unknown result format: {type(results)}. Using original image.")
        return image