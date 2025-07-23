import cv2
import numpy as np
import sys
import time
from ai_edge_litert.interpreter import Interpreter
CONFIDENCE_THRESHOLD = 0.03
IOU_THRESHOLD = 0.5



start = time.perf_counter()
# --- Configuration ---
SEGFORMER_MODEL_PATH = "troof/segformer.tflite"
YOLO_MODEL_PATH = "troof/yolov8s.tflite"
IMAGE_PATH = sys.argv[1]
INPUT_SIZE = (512, 512)

# --- 1. Load Models ---

# Load SegFormer TFLite model and allocate tensors
segformer_interpreter = Interpreter(model_path=SEGFORMER_MODEL_PATH)
segformer_interpreter.allocate_tensors()
segformer_input_details = segformer_interpreter.get_input_details()
segformer_output_details = segformer_interpreter.get_output_details()

# Load the TFLite model and allocate tensors
yolo_interpreter = Interpreter(model_path=YOLO_MODEL_PATH)
yolo_interpreter.allocate_tensors()

# Get input and output tensor details
input_details = yolo_interpreter.get_input_details()
output_details = yolo_interpreter.get_output_details()



# --- 2. Load and Preprocess Image ---

# Load image with OpenCV
original_image = cv2.imread(IMAGE_PATH)

if type(original_image) == None:
    print("ERROR: imagen no encontrada")
    exit(1)

original_height, original_width, _ = original_image.shape
# Resize to the required input size
image_resized = cv2.resize(original_image, INPUT_SIZE)
# Convert from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Prepare image for SegFormer: normalize and add batch dimension
# TFLite models often expect float32 input in [0, 1] range
segformer_input_tensor = image_rgb.astype(np.float32) / 255.0
segformer_input_tensor = np.expand_dims(segformer_input_tensor, axis=0)
segformer_input_tensor = np.transpose(segformer_input_tensor, (0, 3, 1, 2))


# --- 3. Stage 1: Run SegFormer Inference ---

# Set the input tensor, run inference, and get the output
segformer_interpreter.set_tensor(segformer_input_details[0]['index'], segformer_input_tensor)
segformer_interpreter.invoke()
# Output logits shape is typically (1, num_classes, H, W), e.g., (1, 2, 128, 128)
output_logits = segformer_interpreter.get_tensor(segformer_output_details[0]['index'])

# Post-process the segmentation mask
# Get the class with the highest score for each pixel
# Assuming the output is in NCHW format, we find the argmax on the channel axis (axis=1)
mask_predicted = np.argmax(output_logits, axis=1)[0] # Shape becomes (128, 128)

# Upscale the 128x128 mask to 512x512
# Use INTER_NEAREST to avoid creating blurry, intermediate class values
mask_upscaled = cv2.resize(
    mask_predicted.astype(np.uint8),
    INPUT_SIZE,
    interpolation=cv2.INTER_NEAREST
)

# --- 4. Apply Mask to Image for Stage 2 ---

# Create a 3-channel version of the binary mask to multiply with the RGB image
# This will keep pixels where the mask is 1 (building) and zero-out the rest
mask_3d = np.stack([mask_upscaled] * 3, axis=-1)
masked_image_for_yolo = image_rgb * mask_3d

# Set the input tensor, run inference, and get the output
yolo_input_tensor = masked_image_for_yolo.astype(np.float32) / 255.0
yolo_input_tensor = np.expand_dims(yolo_input_tensor, axis=0)
yolo_interpreter.set_tensor(input_details[0]['index'], yolo_input_tensor)
yolo_interpreter.invoke()
output_data = yolo_interpreter.get_tensor(output_details[0]['index']) # Shape: (1, 84, 8400)
output_data = np.squeeze(output_data)

# Lists to store detected boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

for i in range(output_data.shape[1]):  # 8400 detections
    detection = output_data[:, i]  # shape: (84,)    # The first 4 values are box coordinates (cx, cy, w, h)
    # The rest are class scores
    box_coords = detection[:4]
    class_scores = detection[4:]
    
    # Find the class with the highest score
    class_id = np.argmax(class_scores)
    confidence = class_scores[class_id]
    
    # Filter out weak detections
    if confidence > CONFIDENCE_THRESHOLD:
        # Convert model's cx,cy,w,h output to x1,y1,x2,y2 format
        cx, cy, w, h = box_coords
        x1 = int((cx - w / 2) * original_width)
        y1 = int((cy - h / 2) * original_height)
        x2 = int((cx + w / 2) * original_width)
        y2 = int((cy + h / 2) * original_height)
        
        boxes.append([x1, y1, x2 - x1, y2 - y1]) # OpenCV NMS expects (x, y, w, h)
        confidences.append(float(confidence))
        class_ids.append(class_id)


# --- 6. Visualize the Final Results ---

# Apply Non-Maximum Suppression using OpenCV's built-in function
# This removes redundant, overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

# --- 4. Draw Final Bounding Boxes ---

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        
        # Draw the bounding box
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        
        # Prepare the label text
        label = f"Class {class_ids[i]}: {confidences[i]:.2f}"
        
        # Draw the label background
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_resized, (x, y - label_height - 5), (x + label_width, y), (0, 255, 0), -1)
        cv2.putText(image_resized, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Use the plot() method from ultralytics to draw the bounding boxes
# We draw on the original resized image to see the context
final_image_with_boxes = image_resized.copy()

# Create a colored overlay for the segmentation mask (e.g., semi-transparent red)
mask_color = np.zeros_like(final_image_with_boxes)
# Assuming class '1' corresponds to "building"
mask_color[mask_upscaled == 1] = [255, 0, 0] # Red color for buildings (in RGB)

# Blend the original image with the colored mask
# This makes the building areas appear tinted red
final_visualization = cv2.addWeighted(final_image_with_boxes, 1.0, mask_color, 0.6, 0)

# Display or save the final image
output_filename = "predictions/" + "_pred.".join(IMAGE_PATH.split("/")[-1].split("."))
# Convert back to BGR for saving with OpenCV
cv2.imwrite(output_filename, cv2.cvtColor(final_visualization, cv2.COLOR_RGB2BGR))
print(f"✅ Inference complete. Result saved to {output_filename}")
print(f"Finished in {time.perf_counter()-start} seconds")
