import os
import cv2
import numpy as np
from ultralytics import YOLO
import itertools

try:
    from skimage.morphology import dilation, reconstruction, square, skeletonize
    from skimage.util import img_as_bool, img_as_ubyte
    from skimage.graph import route_through_array
except ImportError:
    print("Error: scikit-image not installed.")
    exit()

MODEL_PATH = "runs/segment/train/weights/best.pt" 
VIDEO_PATH = "path to data file" 
OUTPUT_DIR = "crack_detection_output/crack_video"
PIXEL_TO_MM_FACTOR = 0.005
MAX_INTERPOLATION_DISTANCE = 416 

def _find_endpoints_convolution(skeleton_img):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_img.astype(np.float32), -1, kernel)
    neighbor_count = np.round(neighbor_count).astype(np.uint8)
    endpoints_mask = (skeleton_img > 0) & (neighbor_count == 11)
    return np.where(endpoints_mask)

def _create_cost_map(gray_image, skeleton_img):
    cost_map = np.ones_like(gray_image, dtype=np.float32)
    cost_map[skeleton_img == 255] = 0.0001
    cost_map = np.maximum(cost_map, 0.0001)
    return cost_map

def _reconstruct_morphologically(binary_mask, bridge_size=3):
    selem = square(bridge_size)
    bridged_mask = dilation(binary_mask, selem)
    filled_mask_float = reconstruction(binary_mask, bridged_mask, method='dilation')
    return img_as_ubyte(filled_mask_float > 0)

def interpolate_and_predict_cracks_graph(binary_mask, original_gray_image, max_dist_to_connect=75):
    skeleton = skeletonize(img_as_bool(binary_mask))
    skeleton_img = img_as_ubyte(skeleton)
    
    endpoints_coords = _find_endpoints_convolution(skeleton_img // 255)
    endpoints = list(zip(endpoints_coords[0], endpoints_coords[1]))
    
    predicted_paths = []

    if len(endpoints) < 2:
        return binary_mask, predicted_paths

    cost_map = _create_cost_map(original_gray_image, skeleton_img)
    interpolated_mask = np.copy(binary_mask)
    
    for start_point, end_point in itertools.combinations(endpoints, 2):
        dist = np.linalg.norm(np.array(start_point) - np.array(end_point))
        
        if 0 < dist < max_dist_to_connect:
            try:
                indices, weight = route_through_array(
                    cost_map, start=start_point, end=end_point, 
                    geometric=True, fully_connected=True
                )
                path_indices = np.array(indices).T
                interpolated_mask[path_indices[0], path_indices[1]] = 255
                predicted_paths.append(indices)
            except Exception:
                pass 
    
    final_connected_mask = _reconstruct_morphologically(interpolated_mask, bridge_size=5)
    return final_connected_mask, predicted_paths

def analyze_crack_form_risk(angle_deg, length_pixels, width_pixels):
    length_to_width_ratio = length_pixels / width_pixels if width_pixels > 0 else 0
    
    if angle_deg <= 15 or angle_deg >= 165:
        return "Horizontality", "Grade D (Poor)", (0, 0, 255)
    
    elif 15 < angle_deg < 75 or 105 < angle_deg < 165:
        return "Diagonal/Zigzag", "Grade E (Critical)", (0, 0, 139)

    elif 75 <= angle_deg <= 105:
        return "Perpendicular", "Grade B (Good)", (0, 255, 128)
    
    elif length_to_width_ratio < 2 and width_pixels < 20: 
        return "Spiderweb", "Grade A (Excellent)", (0, 255, 0)

    else:
        return "Irregular", "Grade C (Fair)", (0, 165, 255)

def get_crack_properties(mask_data):
    points = np.argwhere(mask_data)
    if len(points) < 2:
        return None, None, None, None, None, None

    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_direction = eigenvectors[:, -1]
    
    angle_rad = np.arctan2(main_direction[0], main_direction[1])
    angle_deg = np.degrees(angle_rad) % 180

    projection = np.dot(points, main_direction)
    length_pixels = np.max(projection) - np.min(projection)
    
    ortho_direction = np.array([-main_direction[1], main_direction[0]])
    ortho_projection = np.dot(points, ortho_direction)
    width_pixels = np.max(ortho_projection) - np.min(ortho_projection)

    length_mm = length_pixels * PIXEL_TO_MM_FACTOR
    width_mm = width_pixels * PIXEL_TO_MM_FACTOR
    if width_mm < 0.05: width_mm = 0.05

    scale_factor = length_pixels / 2
    start_point = (mean - main_direction * scale_factor).astype(int)
    end_point = (mean + main_direction * scale_factor).astype(int)
    start_point_cv = (start_point[1], start_point[0])
    end_point_cv = (end_point[1], end_point[0])

    return angle_deg, width_mm, length_mm, (start_point_cv, end_point_cv), length_pixels, width_pixels

def evaluate_risk(width_mm):
    if width_mm < 0.1:
        return "Grade A (Excellent)", (0, 255, 0)
    elif width_mm < 0.2:
        return "Grade B (Good)", (0, 255, 128)
    elif width_mm < 0.3:
        return "Grade C (Fair)", (0, 165, 255)
    elif width_mm < 0.5:
        return "Grade D (Poor)", (0, 0, 255)
    else:
        return "Grade E (Critical)", (0, 0, 139)

def analyze_cracks():
    try:
        model = YOLO(MODEL_PATH)
    except Exception:
        print(f"Error loading model from {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_name = os.path.basename(VIDEO_PATH)
    output_path = os.path.join(OUTPUT_DIR, f"risk_eval_{video_name}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {VIDEO_PATH}...")

    frame_count = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model.predict(source=img, save=False, conf=0.55, verbose=False)[0]
        
        annotated_img = img.copy()

        if results.masks is not None:
            img_h, img_w = annotated_img.shape[:2]

            combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            masks_data = results.masks.data.cpu().numpy()
            
            for mask_np in masks_data:
                mask_resized = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                combined_mask[mask_resized > 0.5] = 255 

            if np.sum(combined_mask) > 0:
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                final_interpolated_mask, predicted_paths = interpolate_and_predict_cracks_graph(
                    combined_mask, gray_image, max_dist_to_connect=MAX_INTERPOLATION_DISTANCE
                )

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    final_interpolated_mask, 8, cv2.CV_32S
                )

                for path in predicted_paths:
                    for i in range(len(path) - 1):
                        pt1 = (path[i][1], path[i][0])
                        pt2 = (path[i+1][1], path[i+1][0])
                        cv2.line(annotated_img, pt1, pt2, (0, 0, 255), 3)

                for i in range(1, num_labels):
                    mask_binary = (labels == i)
                    area = stats[i, cv2.CC_STAT_AREA]
                    
                    if area < 50: continue

                    properties = get_crack_properties(mask_binary)
                    if properties is None: continue
                    angle_deg, width_mm, length_mm, line_coords, length_pixels, width_pixels = properties
                    
                    width_grade, width_color = evaluate_risk(width_mm)
                    form_type, form_risk_grade, form_color = analyze_crack_form_risk(
                        angle_deg, length_pixels, width_pixels
                    )
                    
                    color = width_color 

                    colored_mask = np.zeros_like(annotated_img)
                    colored_mask[mask_binary] = color
                    annotated_img = cv2.addWeighted(annotated_img, 1.0, colored_mask, 0.4, 0)

                    start_p, end_p = line_coords
                    cv2.arrowedLine(annotated_img, start_p, end_p, (255, 0, 0), 2, tipLength=0.05)
                    cv2.arrowedLine(annotated_img, end_p, start_p, (255, 0, 0), 2, tipLength=0.05)

                    x1 = stats[i, cv2.CC_STAT_LEFT]
                    y1 = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    x2, y2 = x1 + w, y1 + h

                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    info_lines = [
                        f"Crack {i} ({form_type})",
                        f"W/L Risk: {width_grade}",
                        f"Form Risk: {form_risk_grade}",
                        f"W: {width_mm:.2f}mm, L: {length_mm:.1f}mm",
                        f"Ang: {angle_deg:.1f} deg"
                    ]

                    text_x = x1 + 5
                    text_y = y1 + 20
                    
                    for line in info_lines:
                        (w_text, h_text), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_img, (text_x - 2, text_y - h_text - 2), (text_x + w_text + 2, text_y + 2), (255, 255, 255), -1)
                        cv2.putText(annotated_img, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, form_color if 'Risk:' in line else color, 1)
                        text_y += 20

        out.write(annotated_img)
        
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    analyze_cracks()